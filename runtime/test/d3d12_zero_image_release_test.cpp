// A sampled BC texture the title places on its own memory (Xbox device slot 133) waits for memory of its own until
// the title writes it or a command uses it; until then its views name one shared, zeroed image of its exact shape
// (DeviceState::zero_sampled_image). Those images sit in host memory on a discrete GPU and count against a 256 MiB
// cap. One that no placed texture shows any more must go, after the submissions that may still read it, instead of
// staying for the device's life:
//   - Placements of many distinct shapes, one shape twice, each show a shared image (no memory of their own yet), or
//     the test is skipped. The device then keeps one image per shape.
//   - Releasing one of the two textures of a shape keeps that shape's image.
//   - Releasing the rest leaves no shared image and no bytes counted against the cap; after a submission has run and
//     the next one starts, every one of those images has been destroyed.
//   - A placement of a shape seen before gets a shared image again.
// The device has no host image copies, so a placement takes the shared image whatever its driver allows for host-copy
// images. Needs the bindless set (skipped otherwise). Synthetic data only. Set LEONIDA_VK_VALIDATION=1 to run under
// VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <utility>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;  // a copy would release the same interface twice
    Ref& operator=(const Ref&) = delete;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr UINT shapes = 12, first_width = 64, height = 52;  // no other image of the test has this height

// The Vulkan calls the layer makes, seen through the function pointers it loads. The first image made for a shape
// is its shared image: it is made before the placement's own.
PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr real_device_proc = nullptr;
PFN_vkCreateImage real_create_image = nullptr;
PFN_vkDestroyImage real_destroy_image = nullptr;
std::mutex hook_mutex;
std::set<UINT> widths_seen;     // shapes that had an image made
std::set<VkImage> shared_live;  // shared images made and not destroyed yet
unsigned shared_made = 0, shared_destroyed = 0;

VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice device, const VkImageCreateInfo* info,
                                            const VkAllocationCallbacks* allocator, VkImage* image) {
    const VkResult result = real_create_image(device, info, allocator, image);
    if (result == VK_SUCCESS && info->extent.height == height) {
        std::lock_guard lock(hook_mutex);
        if (widths_seen.insert(info->extent.width).second) {
            shared_live.insert(*image);
            ++shared_made;
        }
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice device, VkImage image, const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(hook_mutex);
        if (shared_live.erase(image)) ++shared_destroyed;
    }
    real_destroy_image(device, image, allocator);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkCreateImage") == 0) {
        real_create_image = reinterpret_cast<PFN_vkCreateImage>(real_device_proc(device, name));
        return real_create_image ? reinterpret_cast<PFN_vkVoidFunction>(&create_image) : nullptr;
    }
    if (std::strcmp(name, "vkDestroyImage") == 0) {
        real_destroy_image = reinterpret_cast<PFN_vkDestroyImage>(real_device_proc(device, name));
        return real_destroy_image ? reinterpret_cast<PFN_vkVoidFunction>(&destroy_image) : nullptr;
    }
    return real_device_proc(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) {
        real_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(real_instance_proc(instance, name));
        return real_device_proc ? reinterpret_cast<PFN_vkVoidFunction>(&device_proc) : nullptr;
    }
    return real_instance_proc(instance, name);
}

D3D12_RESOURCE_DESC bc1(UINT w, UINT h) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = w;
    desc.Height = h;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Format = DXGI_FORMAT_BC1_UNORM;
    return desc;
}

// A placement waiting on the shared zero image has no memory of its own yet.
bool shared(ID3D12Resource* texture) { return !rt::d3d12::vulkan_memory_flags(texture); }

}  // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 77;
    real_instance_proc = vk.instance_proc;
    constexpr unsigned slots = shapes + 2;
    void* pages = VirtualAlloc(nullptr, slots * 65536, MEM_RESERVE, PAGE_NOACCESS);
    EXPECT(pages);
    const auto at = [&](unsigned i) { return static_cast<char*>(pages) + i * 65536; };
    int result = 0;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = &instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.bindless = vk.bindless;
        info.bindless_partially_bound = vk.partially_bound;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Ref<ID3D12Resource> textures[shapes], twin, again;
        if (!rt::d3d12::bindless_set_layout(device.p)) {
            std::fprintf(stderr, "SKIP: the device has no bindless set\n");
            result = 77;
        } else if (FAILED(rt::d3d12::create_buffer_at_host_address(device.p, at(0), bc1(first_width, height), 0,
                                                                   IID_PPV_ARGS(&textures[0].p))) ||
                   !shared(textures[0].p)) {
            std::fprintf(stderr, "SKIP: a placed BC texture does not take a shared zero image on this device\n");
            result = 77;
        } else {
            for (UINT i = 1; i < shapes; ++i) {
                EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(
                    device.p, at(i), bc1(first_width + 4 * i, height), 0, IID_PPV_ARGS(&textures[i].p))));
                EXPECT(shared(textures[i].p));
            }
            EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, at(shapes), bc1(first_width, height),
                                                                      0, IID_PPV_ARGS(&twin.p))));
            EXPECT(shared(twin.p));
            const auto placed = rt::d3d12::shared_zero_images(device.p);
            std::printf("%u shapes placed (one twice): %llu shared images, %llu bytes; %u made\n", shapes,
                        static_cast<unsigned long long>(placed.images), static_cast<unsigned long long>(placed.bytes),
                        shared_made);
            EXPECT(placed.images == shapes);
            EXPECT(placed.bytes > 0);
            EXPECT(shared_made == shapes);

            // The shape's other texture still shows its image.
            textures[0].reset();
            const auto one_gone = rt::d3d12::shared_zero_images(device.p);
            EXPECT(one_gone.images == shapes);
            EXPECT(one_gone.bytes == placed.bytes);

            twin.reset();
            for (auto& texture : textures) texture.reset();
            const auto released = rt::d3d12::shared_zero_images(device.p);
            std::printf("all released: %llu shared images, %llu bytes\n",
                        static_cast<unsigned long long>(released.images),
                        static_cast<unsigned long long>(released.bytes));
            EXPECT(released.images == 0);
            EXPECT(released.bytes == 0);

            // The images go through the deferred queue: a submission that has finished, then the next one, which
            // drains what the finished one retired.
            D3D12_COMMAND_QUEUE_DESC queue_desc{};
            queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            Ref<ID3D12CommandQueue> queue;
            Ref<ID3D12CommandAllocator> allocator;
            Ref<ID3D12GraphicsCommandList> list;
            Ref<ID3D12Fence> fence;
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(queue_desc.Type, IID_PPV_ARGS(&allocator.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, queue_desc.Type, allocator.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            EXPECT(list->Close() == S_OK);
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
            for (UINT64 round = 1; round <= 2; ++round) {
                EXPECT(SUCCEEDED(allocator->Reset()));
                EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
                EXPECT(list->Close() == S_OK);
                ID3D12CommandList* lists[] = {list.p};
                queue->ExecuteCommandLists(1, lists);
                EXPECT(SUCCEEDED(queue->Signal(fence.p, round)));
                EXPECT(SUCCEEDED(fence->SetEventOnCompletion(round, nullptr)));
            }
            unsigned destroyed = 0;
            {
                std::lock_guard lock(hook_mutex);
                destroyed = shared_destroyed;
            }
            std::printf("after two submissions: %u of the %u shared images destroyed\n", destroyed, shared_made);
            EXPECT(destroyed == shapes);

            // A shape seen before gets a shared image again.
            {
                std::lock_guard lock(hook_mutex);
                widths_seen.clear();
            }
            EXPECT(SUCCEEDED(rt::d3d12::create_buffer_at_host_address(device.p, at(shapes + 1),
                                                                      bc1(first_width + 4, height), 0,
                                                                      IID_PPV_ARGS(&again.p))));
            const auto later = rt::d3d12::shared_zero_images(device.p);
            std::printf("a shape placed again: %s, %llu shared images\n",
                        shared(again.p) ? "shared" : "with memory of its own",
                        static_cast<unsigned long long>(later.images));
            EXPECT(shared(again.p));
            EXPECT(later.images == 1);
            EXPECT(later.bytes > 0 && later.bytes < placed.bytes);
            again.reset();
            EXPECT(rt::d3d12::shared_zero_images(device.p).bytes == 0);
        }
    }
    VirtualFree(pages, 0, MEM_RELEASE);
    if (vk.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", vk.validation_errors());
        return 1;
    }
    if (!result) std::printf("ok\n");
    return result;
}

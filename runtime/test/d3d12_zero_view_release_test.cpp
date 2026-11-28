// A sampled BC texture the title places on its own memory (Xbox device slot 133) waits for memory of its own until
// the title writes it or a command uses it; until then its views name one shared, zeroed image of its exact shape
// (DeviceState::zero_sampled_image). A copy into it gives it memory of its own in the prepare of the copy's replay,
// and the views of the shared image leave the bindless set there (publish_streamed_texture). A replay in progress
// then may have flushed the set's table with them and submit later, so they must outlive every such replay's
// submissions, not only the submissions made so far:
//   - A placement takes the shared image (skipped otherwise) and a shader-resource view of it makes views of that
//     image.
//   - The copy's replay on the direct queue ends its command buffer, its texture given memory of its own and its
//     submission not recorded yet. From there the compute queue signals a fence, which takes the next retire ticket
//     without waiting for the direct queue's replay, and once that has completed starts an empty replay, which
//     deletes what is due before it waits for the direct queue's replay to submit. No view of the shared image may be
//     destroyed meanwhile: with the ticket of the views' deletion taken when they were forgotten (the submissions so
//     far + 1), the signal's ticket made them due.
//   - Two submissions later every view of the shared image is destroyed: held, not leaked.
// This checks the timing rule only, with the copy's own replay as the one in progress at the forget. That replay
// samples no view of the shared image, so no GPU read of a destroyed view is shown, and no replay of a third queue
// that flushed the table before the forget is built: views released at the end of the copy's replay rather than of the
// last replay in progress would pass too.
// Needs the bindless set (skipped otherwise). Synthetic data only. Set LEONIDA_VK_VALIDATION=1 to run under
// VK_LAYER_KHRONOS_validation.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>

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

constexpr UINT width = 64, height = 52;  // no other image of the test has this height

// The Vulkan calls the layer makes, seen through the function pointers it loads. The first image of the test's height
// is the shared image: it is made before the placement's own.
PFN_vkGetInstanceProcAddr real_instance_proc = nullptr;
PFN_vkGetDeviceProcAddr real_device_proc = nullptr;
PFN_vkCreateImage real_create_image = nullptr;
PFN_vkCreateImageView real_create_view = nullptr;
PFN_vkDestroyImageView real_destroy_view = nullptr;
PFN_vkEndCommandBuffer real_end_command_buffer = nullptr;
std::mutex hook_mutex;
VkImage shared_image = VK_NULL_HANDLE;
std::map<VkImageView, bool> shared_views;  // views of the shared image, not destroyed yet
unsigned views_made = 0, views_destroyed = 0;

unsigned destroyed_views() {
    std::lock_guard lock(hook_mutex);
    return views_destroyed;
}

// A placement waiting on the shared zero image has no memory of its own yet.
bool shared(ID3D12Resource* texture) { return !rt::d3d12::vulkan_memory_flags(texture); }

// The copy's replay ending its command buffer, after its prepare gave the texture memory of its own: the compute
// queue's signal and replay, then up to half a second watching for a view of the shared image being destroyed.
std::atomic<ID3D12Resource*> copy_target{nullptr};  // armed: the texture the copy gives memory of its own
ID3D12CommandQueue* other_queue = nullptr;
ID3D12CommandList* other_list = nullptr;
ID3D12Fence* other_fence = nullptr;
std::thread other_replay;
bool other_signaled = false;
unsigned destroyed_during_copy = 0;

VKAPI_ATTR VkResult VKAPI_CALL end_command_buffer(VkCommandBuffer command_buffer) {
    ID3D12Resource* texture = copy_target.load();
    if (texture && !shared(texture) && copy_target.compare_exchange_strong(texture, nullptr)) {
        const unsigned before = destroyed_views();
        const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
        if (SUCCEEDED(other_queue->Signal(other_fence, 1))) {
            while (std::chrono::steady_clock::now() < until && other_fence->GetCompletedValue() < 1)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            other_signaled = other_fence->GetCompletedValue() >= 1;
        }
        other_replay = std::thread([] {
            ID3D12CommandList* lists[] = {other_list};
            other_queue->ExecuteCommandLists(1, lists);
        });
        const auto watch = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
        while (std::chrono::steady_clock::now() < watch && destroyed_views() == before)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        destroyed_during_copy = destroyed_views() - before;
    }
    return real_end_command_buffer(command_buffer);
}

VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice device, const VkImageCreateInfo* info,
                                            const VkAllocationCallbacks* allocator, VkImage* image) {
    const VkResult result = real_create_image(device, info, allocator, image);
    if (result == VK_SUCCESS && info->extent.height == height) {
        std::lock_guard lock(hook_mutex);
        if (!shared_image) shared_image = *image;
    }
    return result;
}

VKAPI_ATTR VkResult VKAPI_CALL create_view(VkDevice device, const VkImageViewCreateInfo* info,
                                           const VkAllocationCallbacks* allocator, VkImageView* view) {
    const VkResult result = real_create_view(device, info, allocator, view);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(hook_mutex);
        if (shared_image && info->image == shared_image) {
            shared_views[*view] = true;
            ++views_made;
        }
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL destroy_view(VkDevice device, VkImageView view, const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(hook_mutex);
        views_destroyed += unsigned(shared_views.erase(view));
    }
    real_destroy_view(device, view, allocator);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkCreateImage") == 0) {
        real_create_image = reinterpret_cast<PFN_vkCreateImage>(real_device_proc(device, name));
        return real_create_image ? reinterpret_cast<PFN_vkVoidFunction>(&create_image) : nullptr;
    }
    if (std::strcmp(name, "vkCreateImageView") == 0) {
        real_create_view = reinterpret_cast<PFN_vkCreateImageView>(real_device_proc(device, name));
        return real_create_view ? reinterpret_cast<PFN_vkVoidFunction>(&create_view) : nullptr;
    }
    if (std::strcmp(name, "vkDestroyImageView") == 0) {
        real_destroy_view = reinterpret_cast<PFN_vkDestroyImageView>(real_device_proc(device, name));
        return real_destroy_view ? reinterpret_cast<PFN_vkVoidFunction>(&destroy_view) : nullptr;
    }
    if (std::strcmp(name, "vkEndCommandBuffer") == 0) {
        real_end_command_buffer = reinterpret_cast<PFN_vkEndCommandBuffer>(real_device_proc(device, name));
        return real_end_command_buffer ? reinterpret_cast<PFN_vkVoidFunction>(&end_command_buffer) : nullptr;
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

}  // namespace

int main() {
    vktest::Device vk;
    if (!vk.create()) return 77;
    real_instance_proc = vk.instance_proc;
    void* pages = VirtualAlloc(nullptr, 65536, MEM_RESERVE, PAGE_NOACCESS);
    EXPECT(pages);
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
        Ref<ID3D12Resource> texture, upload;
        Ref<ID3D12DescriptorHeap> heap;
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        Ref<ID3D12CommandQueue> compute_queue;  // the other queue, with an empty list and a fence of its own
        Ref<ID3D12CommandAllocator> compute_allocator;
        Ref<ID3D12GraphicsCommandList> compute_list;
        Ref<ID3D12Fence> compute_fence;
        UINT64 round = 0;
        // One list with what `record` puts in it, run to completion.
        const auto submit = [&](auto&& record) {
            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return false;
            record();
            if (list->Close() != S_OK) return false;
            ID3D12CommandList* lists[] = {list.p};
            queue->ExecuteCommandLists(1, lists);
            ++round;
            return SUCCEEDED(queue->Signal(fence.p, round)) && SUCCEEDED(fence->SetEventOnCompletion(round, nullptr));
        };
        const auto empty = [] {};
        if (!rt::d3d12::bindless_set_layout(device.p)) {
            std::fprintf(stderr, "SKIP: the device has no bindless set\n");
            result = 77;
        } else if (FAILED(rt::d3d12::create_buffer_at_host_address(device.p, pages, bc1(width, height), 0,
                                                                   IID_PPV_ARGS(&texture.p))) ||
                   !shared(texture.p)) {
            std::fprintf(stderr, "SKIP: a placed BC texture does not take a shared zero image on this device\n");
            result = 77;
        } else {
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(queue_desc.Type, IID_PPV_ARGS(&allocator.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, queue_desc.Type, allocator.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
            EXPECT(list->Close() == S_OK);
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
            D3D12_COMMAND_QUEUE_DESC compute_desc{};
            compute_desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&compute_desc, IID_PPV_ARGS(&compute_queue.p))));
            EXPECT(SUCCEEDED(device->CreateCommandAllocator(compute_desc.Type,
                                                            IID_PPV_ARGS(&compute_allocator.p))));
            EXPECT(SUCCEEDED(device->CreateCommandList(0, compute_desc.Type, compute_allocator.p, nullptr,
                                                       IID_PPV_ARGS(&compute_list.p))));
            EXPECT(compute_list->Close() == S_OK);
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&compute_fence.p))));
            other_queue = compute_queue.p;
            other_list = compute_list.p;
            other_fence = compute_fence.p;
            D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
            heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heap_desc.NumDescriptors = 1;
            heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));

            // A view of the placement names its shape's shared image while it has no memory of its own.
            device->CreateShaderResourceView(texture.p, nullptr, heap->GetCPUDescriptorHandleForHeapStart());
            unsigned made = 0;
            {
                std::lock_guard lock(hook_mutex);
                made = views_made;
            }
            std::printf("views of the shared image: %u\n", made);
            EXPECT(made > 0);
            EXPECT(shared(texture.p));
            EXPECT(submit(empty));
            EXPECT(destroyed_views() == 0);

            constexpr UINT pitch = 256, rows = height / 4;  // BC1: 8 bytes a 4x4 block, 16 blocks a row
            const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = pitch * rows;
            buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
            buffer.SampleDesc.Count = 1;
            buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                             IID_PPV_ARGS(&upload.p))));
            copy_target.store(texture.p);
            const bool copied = submit([&] {
                D3D12_TEXTURE_COPY_LOCATION to{}, from{};
                to.pResource = texture.p;
                to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                from.pResource = upload.p;
                from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                from.PlacedFootprint.Footprint = {DXGI_FORMAT_BC1_UNORM, width, height, 1, pitch};
                list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
            });
            copy_target.store(nullptr);
            const bool other_ran = other_replay.joinable();
            if (other_ran) other_replay.join();
            EXPECT(copied);
            std::printf("after the copy: %s; while its submission was pending the compute queue %s, and %u of the %u "
                        "views of the shared image were destroyed\n",
                        shared(texture.p) ? "shared" : "with memory of its own",
                        !other_ran ? "did nothing" : other_signaled ? "signaled and replayed" : "replayed unsignaled",
                        destroyed_during_copy, made);
            EXPECT(!shared(texture.p));
            EXPECT(other_ran);  // the copy's replay gave it memory of its own before it ended its command buffer
            EXPECT(other_signaled);
            EXPECT(destroyed_during_copy == 0);

            // Held, not leaked: the copy's replay has recorded its submission and every view goes after it.
            EXPECT(submit(empty));
            EXPECT(submit(empty));
            std::printf("two submissions later: %u of the %u views destroyed\n", destroyed_views(), made);
            EXPECT(destroyed_views() == made);
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

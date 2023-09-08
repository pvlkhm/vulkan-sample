use std::sync::Arc;

use vulkano::{
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    device::physical::PhysicalDevice,
    device::{Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags},
    VulkanLibrary,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo, ColorSpace, PresentMode, CompositeAlpha},
    format::Format, image::{ImageUsage, view::{ImageView, ImageViewCreateInfo}, SwapchainImage, ImageViewType}
};

use winit::{
    event_loop::EventLoop,
    window::WindowBuilder,
    dpi::LogicalSize,
    event::{Event, WindowEvent}
};

use vulkano_win::VkSurfaceBuild;

const WIDTH:  u32 = 800;
const HEIGHT: u32 = 600;

// #[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
// #[cfg(not(debug_assertions))]
// const ENABLE_VALIDATION_LAYERS: bool = false;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_KHRONOS_validation"
];

struct HelloTriangleApplication {
    eloop:      EventLoop<()>,
    queue:      Arc<Queue>,
    swapchain:  Arc<Swapchain>
}

impl HelloTriangleApplication {
    pub fn init() -> Self {
        let library     = VulkanLibrary::new().unwrap();
        let extensions  = Self::get_required_extensions(&library);

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enumerate_portability:  true,
                enabled_extensions:     extensions,
                enabled_layers:         if ENABLE_VALIDATION_LAYERS {
                                            VALIDATION_LAYERS.iter().map(|el| String::from(*el)).collect()
                                        } else { vec![] },
                ..Default::default()
            }
        ).unwrap();

        let physical_device = instance
            .enumerate_physical_devices().unwrap()
            .next().unwrap();

        let (eloop, surface) = Self::init_window(instance);
        let (queue, swapchain) = Self::init_vulkan(surface, physical_device);

        Self {
            eloop,
            queue,
            swapchain
        }
    }

    fn init_window(instance: Arc<Instance>) -> (EventLoop<()>, Arc<Surface>) {
        let eloop  = EventLoop::new();

        let surface = WindowBuilder::new()
            .with_title("Vulkan")
            .with_inner_size(LogicalSize {width: WIDTH, height: HEIGHT})
            .build_vk_surface(&eloop, instance)
            .unwrap();

        (eloop, surface)
    }

    fn init_vulkan(surface: Arc<Surface>, physical_device: Arc<PhysicalDevice>) -> (Arc<Queue>, Arc<Swapchain>) {
        let (gq_idx, pq_idx) = Self::get_queue_family_indexes(&physical_device, &surface);

        let mut device = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions {
                    khr_swapchain: true,
                    ..Default::default()
                },
                enabled_features: Features::empty(),
                queue_create_infos: vec![
                    QueueCreateInfo { // graphic queue + present queue
                        queue_family_index: gq_idx.unwrap(),
                        queues: vec![1.0],
                        ..Default::default()
                    }
                ],
                ..Default::default()
            }
        ).unwrap();

        let queue = device.1.next().unwrap();

        let surface_capabilities = device.0
            .physical_device()
            .surface_capabilities(&surface, Default::default()).unwrap();

        let image_extent = surface_capabilities.current_extent.unwrap();
        let min_image_count = 2;

        let swapchain = Swapchain::new(
            device.0,
            surface,
            SwapchainCreateInfo {
                image_format: Some(Format::B8G8R8A8_SRGB),
                image_color_space: ColorSpace::SrgbNonLinear,
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                present_mode: PresentMode::Fifo,
                pre_transform: surface_capabilities.current_transform,
                composite_alpha: CompositeAlpha::Opaque,
                clipped: true,
                image_extent,
                min_image_count,
                ..Default::default()
            }
        ).unwrap();

        let images      = swapchain.1;
        let swapchain   = swapchain.0;

        (queue, swapchain)
    }

    fn get_required_extensions(library: &VulkanLibrary) -> InstanceExtensions {
        let mut extensions  = vulkano_win::required_extensions(&library);
        if ENABLE_VALIDATION_LAYERS {
            extensions.ext_debug_report = true;
            extensions.ext_debug_utils  = true;
        }

        extensions
    }

    fn get_queue_family_indexes(physical_device: &Arc<PhysicalDevice>, surface: &Arc<Surface>) -> (Option<u32>, Option<u32>) {
        let mut gq_idx = None;
        let mut pq_idx = None;

        for (ii, qf) in physical_device.queue_family_properties().iter().enumerate() {
            if qf.queue_flags.contains(QueueFlags::GRAPHICS) & gq_idx.is_none() { gq_idx = Some(ii as u32) }
            if physical_device.surface_support(ii as u32, surface).unwrap() & pq_idx.is_none() { pq_idx = Some(ii as u32) }
        }

        (gq_idx, pq_idx)
    }

    fn main_loop(self) {
        self.eloop.run(|event, _, flow| {
            flow.set_wait();

            match event {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                    flow.set_exit();
                },
                _ => ()
            }
        });
    }
}

fn main() {
    let app = HelloTriangleApplication::init();
    app.main_loop();
}

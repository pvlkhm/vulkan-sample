use std::sync::Arc;

use vulkano::{
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    device::physical::PhysicalDevice,
    device::{Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags},
    VulkanLibrary,
    swapchain::Surface
};

use winit::{
    event_loop::EventLoop,
    window::{WindowBuilder, Window},
    dpi::LogicalSize,
    event::{Event, WindowEvent}
};

use vulkano_win::VkSurfaceBuild;

const WIDTH:  u32 = 800;
const HEIGHT: u32 = 600;

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_KHRONOS_validation"
];

struct HelloTriangleApplication {
    eloop:              EventLoop<()>,
    window:             Arc<Surface>,
    device:             Arc<Device>,
    graphic_queue:      Arc<Queue>
}

impl HelloTriangleApplication {
    pub fn init() -> Self {
        let (instance, device, graphic_queue) = Self::init_vulkan();
        let (eloop, window) = Self::init_window(instance);

        Self {
            eloop,
            window,
            device,
            graphic_queue
        }
    }

    fn init_window(instance: Arc<Instance>) -> (EventLoop<()>, Arc<Surface>) {
        let eloop  = EventLoop::new();

        let window = WindowBuilder::new()
            .with_title("Vulkan")
            .with_inner_size(LogicalSize {width: WIDTH, height: HEIGHT})
            .build_vk_surface(&eloop, instance)
            .unwrap();

        (eloop, window)
    }

    fn init_vulkan() -> (Arc<Instance>, Arc<Device>, Arc<Queue>) {
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

        let queue_family_index = Self::get_queue_family_index(&physical_device).unwrap();

        let mut device = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: DeviceExtensions::empty(),
                enabled_features: Features::empty(),
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: queue_family_index as u32,
                    queues: vec![1.0],
                    ..Default::default()
                }],
                ..Default::default()
            }
        ).unwrap();

        let graphic_queue = device.1.next().unwrap();

        (instance, device.0, graphic_queue)
    }

    fn get_required_extensions(library: &VulkanLibrary) -> InstanceExtensions {
        let mut extensions  = vulkano_win::required_extensions(&library);
        if ENABLE_VALIDATION_LAYERS {
            extensions.ext_debug_report = true;
            extensions.ext_debug_utils  = true;
        }

        extensions
    }

    fn get_queue_family_index(physical_device: &Arc<PhysicalDevice>) -> Option<usize> {
        for (ii, qf) in physical_device.queue_family_properties().iter().enumerate() {
            if qf.queue_flags.contains(QueueFlags::GRAPHICS) { return Some(ii) }
        }

        return None
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

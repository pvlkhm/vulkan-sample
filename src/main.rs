use std::sync::Arc;
use vulkano_win;
use vulkano::{
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    VulkanLibrary
};

use winit::{
    event_loop::EventLoop,
    window::{WindowBuilder, Window},
    dpi::LogicalSize,
    event::{Event, WindowEvent}
};

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
    eloop:      EventLoop<()>,
    window:     Window,
    instance:   Arc<Instance>
}

impl HelloTriangleApplication {
    pub fn init() -> Self {
        let (eloop, window) = Self::init_window();
        let instance = Self::init_vulkan();

        Self {
            eloop,
            window,
            instance
        }
    }

    fn init_window() -> (EventLoop<()>, Window) {
        let eloop  = EventLoop::new().unwrap();
        let window = WindowBuilder::new()
            .with_title("Vulkan")
            .with_inner_size(LogicalSize {width: WIDTH, height: HEIGHT})
            .build(&eloop).unwrap();

        (eloop, window)
    }

    fn init_vulkan() -> Arc<Instance> {
        let library     = VulkanLibrary::new().unwrap();
        let extensions  = Self::get_required_extensions(&library);

        Instance::new(
            library,
            InstanceCreateInfo {
                enumerate_portability:  true,
                enabled_extensions:     extensions,
                enabled_layers:         if ENABLE_VALIDATION_LAYERS {
                                            VALIDATION_LAYERS.iter().map(|el| String::from(*el)).collect()
                                        } else { vec![] },
                ..Default::default()
            }
        ).unwrap()
    }

    fn get_required_extensions(library: &VulkanLibrary) -> InstanceExtensions {
        let mut extensions  = vulkano_win::required_extensions(&library);
        if ENABLE_VALIDATION_LAYERS {
            extensions.ext_debug_report = true;
            extensions.ext_debug_utils  = true;
        }

        extensions
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
        }).unwrap();
    }
}

fn main() {
    let app = HelloTriangleApplication::init();
    app.main_loop();
}

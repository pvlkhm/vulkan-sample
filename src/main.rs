use std::{sync::Arc, future::Future};

use vulkano::{
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    device::physical::PhysicalDevice,
    device::{Device, DeviceCreateInfo, DeviceExtensions, Features, Queue, QueueCreateInfo, QueueFlags},
    VulkanLibrary,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo, ColorSpace, PresentMode, CompositeAlpha, acquire_next_image, self, SwapchainPresentInfo},
    format::{Format, ClearValue},
    image::{ImageUsage, SwapchainImage, view::{ImageView, ImageViewCreateInfo}, ImageViewType, ImageSubresourceRange, ImageLayout, ImageAspects},
    pipeline::{graphics::{GraphicsPipeline, input_assembly::{PrimitiveTopology, InputAssemblyState}, rasterization::{RasterizationState, PolygonMode}, color_blend::ColorBlendState, render_pass::{PipelineRenderPassType, PipelineRenderingCreateInfo}, viewport::{ViewportState, Viewport, Scissor}}, PartialStateMode}, render_pass::{Subpass, Framebuffer, FramebufferCreateInfo, RenderPass, RenderPassCreateInfo, AttachmentDescription, SubpassDescription, AttachmentReference, SubpassDependency}, command_buffer::{pool::{CommandPool, CommandPoolCreateInfo, CommandBufferAllocateInfo}, CommandBufferLevel, AutoCommandBufferBuilder, allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo}, CommandBufferUsage, RenderPassBeginInfo, SubpassContents, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, synced::SyncCommandBufferBuilder}, NonExhaustive, sync::{GpuFuture, PipelineStages, AccessFlags, DependencyFlags}
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

mod shader_vert {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: r"
            #version 450

            layout(location = 0) out vec3 fragColor;

            vec2 positions[3] = vec2[](
                vec2(0.0, -0.5),
                vec2(0.5, 0.5),
                vec2(-0.5, 0.5)
            );

            vec3 colors[3] = vec3[](
                vec3(1.0, 0.0, 0.0),
                vec3(1.0, 1.0, 0.0),
                vec3(0.0, 1.0, 1.0)
            );

            void main() {
                gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
                fragColor = colors[gl_VertexIndex];
            }
        "
    }
}

mod shader_frag {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: r"
            #version 450

            layout(location = 0) in vec3 fragColor;

            layout(location = 0) out vec4 outColor;

            void main() {
                outColor = vec4(fragColor, 1.0);
            }
        "
    }
}

struct HelloTriangleApplication {
    eloop:          EventLoop<()>,
    device:         Arc<Device>,
    queue:          Arc<Queue>,
    swapchain:      Arc<Swapchain>,
    pipe:           Arc<GraphicsPipeline>,
    framebuffers:   Vec<Arc<Framebuffer>>,
    cbs:            Vec<Arc<PrimaryAutoCommandBuffer>>
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
        let (queue, swapchain, images, device) = Self::init_vulkan(surface, physical_device);
        let (pipe, framebuffers, cbs) = Self::init_pipe(device.clone(), swapchain.clone(), images, queue.clone());

        Self {
            eloop,
            device,
            queue,
            swapchain,
            pipe,
            framebuffers,
            cbs
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

    fn init_vulkan(surface: Arc<Surface>, physical_device: Arc<PhysicalDevice>)
        -> (Arc<Queue>, Arc<Swapchain>, Vec<Arc<ImageView<SwapchainImage>>>, Arc<Device>) {
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
            device.0.clone(),
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

        let images = swapchain.1.iter().map(|sc_image| {
            ImageView::new(sc_image.clone(), ImageViewCreateInfo {
                view_type:          ImageViewType::Dim2d,
                format:             Some(swapchain.0.clone().image_format()),
                subresource_range:  ImageSubresourceRange {
                    aspects: ImageAspects::COLOR,
                    array_layers: std::ops::Range { start: 0, end: 1 },
                    mip_levels: std::ops::Range { start: 0, end: 1 }
                },
                ..Default::default()
            }).unwrap()
        }).collect();
        let swapchain   = swapchain.0;

        (queue, swapchain, images, device.0)
    }

    fn init_pipe(device: Arc<Device>, swapchain: Arc<Swapchain>, images: Vec<Arc<ImageView<SwapchainImage>>>, queue: Arc<Queue>)
                -> (Arc<GraphicsPipeline>, Vec<Arc<Framebuffer>>, Vec<Arc<PrimaryAutoCommandBuffer>>) {
        let shader_vert = shader_vert::load(device.clone()).unwrap();
        let shader_frag = shader_frag::load(device.clone()).unwrap();

        let render_pass = RenderPass::new(device.clone(), RenderPassCreateInfo {
            attachments: vec![
                AttachmentDescription {
                    format: Some(swapchain.image_format()),
                    samples: vulkano::image::SampleCount::Sample1,
                    load_op: vulkano::render_pass::LoadOp::Clear,
                    store_op: vulkano::render_pass::StoreOp::Store,
                    stencil_load_op: vulkano::render_pass::LoadOp::DontCare,
                    stencil_store_op: vulkano::render_pass::StoreOp::DontCare,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::PresentSrc,
                    ..Default::default()
                }
            ],
            subpasses: vec![
                SubpassDescription {
                    color_attachments: vec![
                        Some(AttachmentReference {
                            attachment: 0,
                            layout: ImageLayout::ColorAttachmentOptimal,
                            ..Default::default()
                        })
                    ],
                    ..Default::default()
                }
            ],
            dependencies: vec![
                SubpassDependency {
                    src_subpass: None,
                    dst_subpass: Some(0),
                    src_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                    src_access: AccessFlags::empty(),
                    dst_stages: PipelineStages::COLOR_ATTACHMENT_OUTPUT,
                    dst_access: AccessFlags::COLOR_ATTACHMENT_WRITE,
                    ..Default::default()
                }
            ],
            ..Default::default()
        }).unwrap();

        let pipe = GraphicsPipeline::start()
            .vertex_shader(shader_vert.entry_point("main").unwrap(), ())
            .fragment_shader(shader_frag.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState {
                topology: PartialStateMode::Fixed(PrimitiveTopology::TriangleList),
                ..Default::default()
            })
            .viewport_state(ViewportState::Fixed {
                data: vec![(
                    Viewport {
                        origin:     [0.0, 0.0],
                        dimensions: [swapchain.image_extent()[0] as f32, swapchain.image_extent()[1] as f32],
                        depth_range: std::ops::Range { start: 0.0, end: 1.0 }
                    },
                    Scissor {
                        origin:     [0, 0],
                        dimensions: swapchain.image_extent()
                    }
                )]
            })
            .rasterization_state(RasterizationState {
                polygon_mode: PolygonMode::Fill,
                ..Default::default()
            })
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone()).unwrap();

        let frambuffers: Vec<Arc<Framebuffer>> = images.iter().map(|image| {
            Framebuffer::new(render_pass.clone(), FramebufferCreateInfo {
                attachments:    vec![image.clone()],
                extent:         swapchain.image_extent(),
                ..Default::default()
            }).unwrap()
        }).collect();

        let allocator = StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                primary_buffer_count:   1,
                secondary_buffer_count: 0,
                ..Default::default()
            }
        );

        let cbs = frambuffers.iter().map(|fb| {
            let mut cb_builder = AutoCommandBufferBuilder::primary(
                &allocator,
                queue.queue_family_index(),
                CommandBufferUsage::SimultaneousUse
            ).unwrap();

            cb_builder
                .begin_render_pass(RenderPassBeginInfo {
                    clear_values: vec![Some(ClearValue::Float([0.0,0.0,0.0,1.0]))],
                    ..RenderPassBeginInfo::framebuffer(
                        fb.clone(),
                    )
                }, SubpassContents::Inline).unwrap()
                .bind_pipeline_graphics(pipe.clone())
                .draw(3, 1, 0, 0).unwrap()
                .end_render_pass().unwrap();

            Arc::new(cb_builder.build().unwrap())
        }).collect();

        (pipe, frambuffers, cbs)
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
        self.eloop.run(move |event, _, flow| {
            flow.set_poll();

            let (idx, _, image_acq) = acquire_next_image(self.swapchain.clone(), None).unwrap();

            let future = image_acq
                    .then_execute(self.queue.clone(), self.cbs[idx as usize].clone()).unwrap()
                    .then_swapchain_present(self.queue.clone(), SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), idx))
                    .then_signal_fence_and_flush();

            future.unwrap().wait(None).unwrap();

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

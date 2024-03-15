extern crate image;

use graphics::clear;
use ml_library::network::Network;
use opengl_graphics::GlGraphics;
use piston::*;
use glutin_window::*;
use image::*;
use piston_window::*;
use rusttype::Font;
use memory_stats::memory_stats;
use std::thread::{self, Thread};
use std::time::Duration;

use crate::{section::Section, widget::WidgetType};

const PADDING: f64 = 10.0;
const HEADER: f64 = 70.0;

pub struct GUI<'a> {
    pub window: PistonWindow,
    pub gl: GlGraphics,
    pub sections: Vec<Section>,
    pub nn: Network,
    pub data: Vec<[Vec<f64>; 2]>,
    pub epochs_per_second: usize,
    pub epochs: usize,
    pub image_data: Vec<Vec<u8>>,
    pub font: Font<'a>,
    pub model_name: String,
    pub x_range: [f64; 2]
}

impl GUI<'_> {
    pub fn new(nn: Network) -> Self {
        let window = WindowSettings::new("GUI", [720, 480])
        .exit_on_esc(true)
        .build()
        .unwrap();

        let font = Font::try_from_bytes(
            include_bytes!("../assets/BebasNeue-Regular.ttf")).unwrap();

        GUI {
            window,
            gl: GlGraphics::new(OpenGL::V3_2),
            sections: vec![],
            nn,
            data: vec![],
            epochs_per_second: 1,
            epochs: 0,
            image_data: vec![],
            font,
            model_name: "Model".to_string(),
            x_range: [-1.0, 1.0]
        }
    }

    pub fn set_sections(&mut self, sections: Vec<Vec<WidgetType>>) {
        let section_count = sections.len();
        let section_width = self.window.draw_size().width / section_count as f64;
        let section_height = self.window.draw_size().height - HEADER;

        let mut sects = vec![];
        for i in 0..section_count {
            let coords1 = [section_width * i as f64, HEADER];
            let coords = [coords1[0] + PADDING,
                coords1[1] + PADDING,
                coords1[0] - PADDING,
                coords1[1] - PADDING
            ];
            sects.push(Section::new(coords, section_width, section_height));
            sects[i].set_widgets(&sections[i]);
        }
        self.sections = sects;
    }

    pub fn set_model_name(&mut self, name: &str) {
        self.model_name = name.to_string();
    }

    pub fn set_epochs_per_second(&mut self, epochs: usize) {
        self.epochs_per_second = epochs;
    }

    pub fn render(&mut self, evts: &Event, args: RenderArgs, window_ctx: &mut G2dTextureContext) {
        let mut glyphs = self.window.load_font("assets/BebasNeue-Regular.ttf").unwrap();

        self.window.draw_2d(evts, |ctx, gl, device| {
            clear([0.3, 0.3, 0.3, 1.0], gl);

            let _ = text::Text::new_color([1.0, 1.0, 1.0, 1.0], 20).draw(
                &format!("Cost: {}", self.nn.cost),
                &mut glyphs,
                &ctx.draw_state,
                ctx.transform.trans(PADDING * 4.0, PADDING * 4.0), gl
            );

            let _ = text::Text::new_color([1.0, 1.0, 1.0, 1.0], 20).draw(
                &format!("Epochs: {}", self.epochs),
                &mut glyphs,
                &ctx.draw_state,
                ctx.transform.trans(PADDING * 30.0, PADDING * 4.0), gl
            );

            let _ = text::Text::new_color([1.0, 1.0, 1.0, 1.0], 20).draw(
                &format!("Batch Size: {}", self.nn.batch_size),
                &mut glyphs,
                &ctx.draw_state,
                ctx.transform.trans(PADDING * 45.0, PADDING * 4.0), gl
            );

            glyphs.factory.encoder.flush(device);

            for i in 0..self.sections.len() {
                self.sections[i].render(ctx, gl, window_ctx);
            }
        });
    }

    pub fn set_data(&mut self, data: Vec<[Vec<f64>; 2]>) {
        self.data = data;
        let expected_img = self.get_expected_img();
        for i in 0..self.sections.len() {
            for j in 0..self.sections[i].widgets.len() {
                let widget = &mut self.sections[i].widgets[j];
                if widget.widget_type == WidgetType::OutputImg {
                    widget.expected_image = expected_img.clone();
                }
            }
        }
    }

    pub fn set_cost_expiration(&mut self, expire: bool, epochs: usize) {
        for i in 0..self.sections.len() {
            for j in 0..self.sections[i].widgets.len() {
                self.sections[i].widgets[j].set_cost_expiration(expire, epochs);
            }
        }
    }

    pub fn pop_cost(&mut self) {
        for i in 0..self.sections.len() {
            for j in 0..self.sections[i].widgets.len() {
                self.sections[i].widgets[j].pop_cost();
            }
        }
    }

    pub fn run(&mut self) {
        let mut events = Events::new(EventSettings::new()).ups(60);
        // thread::spawn(|| loop {
        //     // Get memory stats
        //     if let Some(usage) = memory_stats() {
        //         println!("Physical memory usage: {}", usage.physical_mem / 100_000);
        //         println!("Virtual memory usage: {}", usage.virtual_mem / 100_000);
        //     } else {
        //         println!("Couldn't get the current memory usage :(");
        //     }
    
        //     // Sleep for a second
        //     thread::sleep(Duration::from_secs(1));
        // }); 
        while let Some(e) = events.next(&mut self.window) {
            if let Some(args) = e.render_args() {
                let window_ctx = &mut self.window.create_texture_context();
                self.render(&e, args, window_ctx);
            }

            if let Some(args) = e.update_args() {
                self.nn.train(self.data.clone(), self.epochs_per_second);
                self.image_data = self.get_network_img();
                self.epochs += self.epochs_per_second;
                for i in 0..self.sections.len() {
                    let weights = self.nn.get_weights();
                    let biases = self.nn.get_biases();
                    let nodes = self.nn.get_nodes();
                    let network_graph = self.get_network_graph();
                    self.sections[i].set_architecture((weights, biases, nodes));
                    self.sections[i].update(
                        self.nn.cost, 
                        self.epochs_per_second, 
                        &self.image_data,
                        network_graph
                    );
                }
            }

            if let Some(Button::Keyboard(key)) = e.press_args() {
                match key {
                    Key::F => 
                        for i in 0..self.data.len() {
                            println!("------------------------\n{i}) Input: {:?} Output: {:?} Target: {:?}",
                                self.data[i][0], 
                                self.nn.forward(self.data[i][0].clone()), 
                                self.data[i][1]
                            );
                        },
                    Key::I => 
                        self.save_img(),
                    Key::R => 
                        self.restart(),
                    Key::Backspace =>
                        self.pop_cost(),
                    Key::S => 
                        {
                            self.nn.save_model(&self.model_name);
                            println!("{} Saved Succesfully!", self.model_name);
                        },
                    Key::L =>
                        {
                            self.restart();
                            self.nn.load_model(&self.model_name);
                            println!("{} Loaded Succesfully!", self.model_name);
                        },
                    _ => 
                        println!("No Function Associated With That Button"),
                }
            }
        }
    }

    fn restart(&mut self) {
        for i in 0..self.sections.len() {
            self.sections[i].cost = 0.0;
            for j in 0..self.sections[i].widgets.len() {
                let widget = &mut self.sections[i].widgets[j];
                widget.cost = vec![];
                widget.epochs = 0;
            }
        }
        self.epochs = 0;
        self.nn.reset();
    }

    fn get_network_graph(&mut self) -> Vec<f64> {
        let increment = 0.01;
        let mut counter = self.x_range[0];
        let mut outputs = vec![];

        while counter <= self.x_range[1] {
            outputs.push(self.nn.forward(vec![counter])[0]);
            counter += increment;
        }
        outputs
    }

    fn get_network_img(&mut self) -> Vec<Vec<u8>> {

        let width = 28;
        let height = 28;
        
        let mut new_image: Vec<Vec<u8>> = vec![vec![0; width]; height];

        for y in 0..height {
            for x in 0..width {
                let x_coord = x as f64 / (width - 1) as f64;
                let y_coord = y as f64 / (height - 1) as f64;
                let inputs = vec![x_coord, y_coord];                
                let int = (self.nn.forward(inputs)[0] * 255.0) as u8;
                new_image[y][x] = int;
            }
        }  
        new_image
    }

    fn get_expected_img(&mut self) -> Vec<Vec<u8>> {
        
        let mut new_image: Vec<Vec<u8>> = vec![vec![0; 28]; 28];
        if self.data.len() == 784 {
            let mut index = 0;

            for y in 0..new_image.len() as usize {
                for x in 0..new_image[y].len() as usize {
                    let int = self.data[index][1][0] * 255.0;
                    new_image[y][x] = int as u8;
                    index += 1;
                }
            }  
        }
        new_image
    }

    fn save_img(&mut self) { 
        let image = self.get_network_img();

        let mut img = RgbImage::new(28, 28);

        for y in 0..img.dimensions().1 as usize {
            for x in 0..img.dimensions().0 as usize {
                let int = image[y][x];
                let rgb = Rgb([int, int, int]);
                img.put_pixel(x as u32, y as u32, rgb);
            }
        }  

        let _  = img.save("Output.png").unwrap();
        println!("Saved Image");
    }
}
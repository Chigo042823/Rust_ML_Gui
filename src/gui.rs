use graphics::clear;
use ml_library::network::Network;
use opengl_graphics::GlGraphics;
use piston::*;
use glutin_window::*;
use image::*;

use crate::{section::Section, widget::WidgetType};

const PADDING: f64 = 10.0;

pub struct GUI {
    pub window: GlutinWindow,
    pub gl: GlGraphics,
    pub sections: Vec<Section>,
    pub nn: Network,
    pub data: Vec<[Vec<f64>; 2]>
}

impl GUI {
    pub fn new(nn: Network) -> Self {
        let window = WindowSettings::new("GUI", [720, 480])
        .graphics_api(OpenGL::V3_2)
        .exit_on_esc(true)
        .build()
        .unwrap();

        GUI {
            window,
            gl: GlGraphics::new(OpenGL::V3_2),
            sections: vec![],
            nn,
            data: vec![]
        }
    }

    pub fn set_sections(&mut self, sections: Vec<Vec<WidgetType>>) {
        let section_count = sections.len();
        let section_width = self.window.draw_size().width / section_count as f64;
        let section_height = self.window.draw_size().height;

        let mut sects = vec![];
        for i in 0..section_count {
            let coords1 = [section_width * i as f64, 0.0];
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

    pub fn render(&mut self, args: RenderArgs) {
        const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

        self.gl.draw(args.viewport(), |ctx, gl| {
            clear(BLACK, gl);

            for i in 0..self.sections.len() {
                self.sections[i].render(ctx, gl);
            }
        });
    }

    pub fn set_data(&mut self, data: Vec<[Vec<f64>; 2]>) {
        self.data = data;
    }

    pub fn run(&mut self) {
        let mut events = Events::new(EventSettings::new());
        while let Some(e) = events.next(&mut self.window) {
            if let Some(args) = e.render_args() {
                self.render(args);
            }

            if let Some(args) = e.update_args() {
                self.nn.train(self.data.clone(), 10);
                for i in 0..self.sections.len() {
                    let weights = self.nn.get_weights();
                    let biases = self.nn.get_biases();
                    let nodes = self.nn.get_nodes();
                    self.sections[i].set_architecture((weights, biases, nodes));
                    self.sections[i].update(&mut self.gl, self.nn.cost);
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
                    Key::S => 
                        self.save_img(),
                    _ => todo!(),
                }
            }
        }
    }
    fn save_img(&mut self) {
        let mut new_image = RgbImage::new(28, 28);

        for y in 0..new_image.dimensions().1 as usize {
            for x in 0..new_image.dimensions().0 as usize {
                let int = (self.nn.forward(vec![x as f64, y as f64])[0] * 255.0) as u8;
                let rgb = Rgb([int, int, int]);
                new_image.put_pixel(x as u32, y as u32, rgb);
                // if int > 10 {
                //     print!("#");
                // } else {
                //     print!(" ");
                // }
            }
        // println!();
        }   

        let _  = new_image.save("Output.png").unwrap();
        println!("Saved Image");
    }
}
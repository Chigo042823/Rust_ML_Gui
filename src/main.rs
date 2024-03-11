use ml_gui::{gui::GUI, section::Section, widget::WidgetType};
use ml_library::{layer::{DenseLayer, Layer}, network::Network};
use image::*;
use WidgetType::*;

fn main() {
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(DenseLayer::new(2, 7, ml_library::activation::ActivationFunction::Sigmoid)),
        Box::new(DenseLayer::new(7, 7, ml_library::activation::ActivationFunction::Sigmoid)),
        Box::new(DenseLayer::new(7, 1, ml_library::activation::ActivationFunction::Sigmoid)),
    ];
    let nn = Network::new(layers, 0.3, 12);
    let sections = vec![
        vec![OutputIMG]
    ];

    let xor_data: Vec<[Vec<f64>; 2]> = vec![
        [vec![1.0, 0.0], vec![0.0]],
        [vec![0.0, 0.0], vec![1.0]],
        [vec![1.0, 1.0], vec![1.0]],
        [vec![0.0, 1.0], vec![0.0]],
    ]; 

    let mut app = GUI::new(nn);
    app.set_sections(sections);
    app.set_epochs_per_second(30);
    app.set_cost_expiration(false, 5);
    // app.set_data(xor_data);
    // app.run();
    digit_model(app);
}

pub fn digit_model(mut app: GUI) {
    let mut data: Vec<[Vec<f64>; 2]> = vec![];

    let img = image::open("mnist_8.png").unwrap();

    let dims = img.dimensions();

    for y in 0..dims.1 {
        for x in 0..dims.0 {
            let pixel = img.get_pixel(x, y).0;
            let intensity = (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3);
            let inputs = vec![x as f64 / 27 as f64, y as f64 / 27 as f64];
            // let inputs = vec![x as f64, y as f64];
            data.push([inputs, vec![(intensity as f64 / 255.0)]]);
        }
    }

    app.set_data(data);
    app.run();
}


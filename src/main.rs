use ml_gui::{gui::GUI, widget::WidgetType};
use ml_library::{layer::Layer, layer::LayerType::Dense, network::Network};
use image::*;
use WidgetType::*;

fn main() {
    let layers: Vec<Layer> = vec![
        Layer::new(2, 10, Dense, ml_library::activation::ActivationFunction::Sigmoid),
        Layer::new(10, 10, Dense, ml_library::activation::ActivationFunction::Sigmoid),
        Layer::new(10, 1, Dense, ml_library::activation::ActivationFunction::Sigmoid),
    ];

    let nn = Network::new(layers, 0.02, 2);
    let sections: Vec<Vec<WidgetType>> = vec![
        vec![OutputImg, CostPlot], 
        vec![Architecture], 
    ];

    // nn.load_model("8Model");

    let mut app = GUI::new(nn);
    app.set_sections(sections);
    app.set_epochs_per_second(50);
    app.set_cost_expiration(false, 20);
    app.set_model_name("6Model");
    app.x_range = [-10.0, 10.0];
    // sin_model(&mut app);
    digit_model(app);
    // xor_model(&mut app);
    // upscale_img(&mut nn, [500, 500]);
}

pub fn xor_model(app: &mut GUI) {

    let data: Vec<[Vec<f64>; 2]> = vec![
        [vec![1.0, 0.0], vec![0.0]],
        [vec![0.0, 0.0], vec![1.0]],
        [vec![1.0, 1.0], vec![1.0]],
        [vec![0.0, 1.0], vec![0.0]],
    ]; 

    app.set_data(data);
    app.run();
}

pub fn digit_model(mut app: GUI) {
    let mut data: Vec<[Vec<f64>; 2]> = vec![];

    let img = image::open("mnist_6.png").unwrap();

    let dims = img.dimensions();

    for y in 0..dims.1 {
        for x in 0..dims.0 {
            let pixel = img.get_pixel(x, y).0;
            let intensity = (pixel[0] / 3) + (pixel[1] / 3) + (pixel[2] / 3);
            let x_coord = x as f64 / (dims.0 - 1) as f64;
            let y_coord = y as f64 / (dims.1 - 1) as f64;
            let inputs = vec![x_coord, y_coord];
            data.push([inputs, vec![(intensity as f64 / 255.0)]]);
        }
    }

    app.set_data(data);
    app.run();
}

pub fn sin_model(app: &mut GUI) {

    let mut func_outputs = vec![];
    let increment = 0.1;
    let mut counter = app.x_range[0];

    while counter < app.x_range[1] {
        func_outputs.push([vec![counter.clone()], vec![func(&counter)]]);
        counter += increment;
    }

    app.set_data(func_outputs);
    app.run()
}

fn func(x: &f64) -> f64 {
    x.sin()
}

fn upscale_img(nn: &mut Network, dims: [u32; 2]) {

    let output_image = ImageBuffer::from_fn(dims[0], dims[1], |x, y| {
        let x_coord = x as f64 / (dims[0] - 1) as f64;
        let y_coord = y as f64 / (dims[1] - 1) as f64;
        let inputs = vec![x_coord, y_coord];
        let pix = (nn.forward(inputs)[0] * 255.0) as u8;
        Rgba([pix, pix, pix, 255]) // Varying colors in a gradient
    });
            

    let _  = output_image.save("Output.png").unwrap();
    println!("Saved Image");
}
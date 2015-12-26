extern crate cgmath;

use std::ops::{Index, IndexMut, Div};
use std::io::Write;
use std::iter;
use std::default::Default;
use std::f32::consts::PI;

use cgmath::EuclideanVector;
use cgmath::Vector;

use std::convert::From;

type Vector2 = cgmath::Vector2<f32>;
type Vector3 = cgmath::Vector3<f32>;

#[derive(Clone, Copy)]
struct Ray {
    pub origin: Point3,
    pub direction: Vector3,
}

#[derive(Default, Clone, Copy)]
struct Color3 {
    pub r: f32,
    pub b: f32,
    pub g: f32,
}

impl Div<f32> for Color3 {
    type Output = Color3;

    fn div(self, rhs: f32) -> Color3 {
        Color3 {
            r: self.r / rhs,
            b: self.b / rhs,
            g: self.g / rhs,
        }
    }
}

impl From<Vector3> for Color3 {
    fn from(v: Vector3) -> Color3 {
        Color3 {
            r: v.x,
            g: v.y,
            b: v.z,
        }
    }
}

impl Color3 {
    fn vec(self) -> Vector3 {
        Vector3 {
            x: self.r,
            y: self.g,
            z: self.b,
        }
    }
}

struct Triangle
{
    pub vertex: [Point3; 3],
    pub normal: [Vector3; 3],
    pub color: Color3,
}

impl Triangle {
    fn evaluate_finite_scattering_density(&self, _w_i: &Vector3, _w_o: &Vector3) -> Color3 {
        self.color / PI
    }
}

struct Light {
    pub position: Point3,
    pub power: Point3,
}

struct Scene {
    pub triangles: Vec<Triangle>,
    pub lights: Vec<Light>,
}

struct Camera {
    pub z_near: f32,
    pub z_far: f32,
    pub fov_x: f32,
}

impl Default for Camera {
    fn default() -> Camera {
        Camera {
            z_near: -0.1,
            z_far: -100.0,
            fov_x: PI / 2.,
        }
    }
}

type Radiance3 = Color3;
type Power3 = Color3;
type Point2 = Vector2;
type Point3 = Vector3;

struct Image {
    width: usize,
    height: usize,
    data: Vec<Radiance3>,
}

fn ppm_gamma_encode(radiance: f32, d: f32) -> i32 {
    (1f32.min(0f32.max(radiance * d)).powf(1f32 / 2.2) * 255f32) as i32
}

impl Image {
    fn new(width: usize, height: usize) -> Image {
        Image {
            width: width,
            height: height,
            data: iter::repeat(Radiance3 { r: 0., g: 0., b: 0. }).take(width * height).collect(),
        }
    }

    fn width(&self) -> usize { self.width }
    fn height(&self) -> usize { self.height }

    fn save(&self, filename: &str, d: f32) -> std::io::Result<()> {
        use std::fs::File;

        let mut file = try!(File::create(filename));

        try!(writeln!(file, "P3 {} {} 255", self.width, self.height));
        for y in 0..self.height {
            try!(writeln!(file, "\n# y = {}", y));
            for x in 0..self.width {
                let ref rad = self[(x, y)];
                try!(writeln!(file, "{} {} {}",
                              ppm_gamma_encode(rad.r, d),
                              ppm_gamma_encode(rad.g, d),
                              ppm_gamma_encode(rad.b, d)));
            }
        }
        
        Ok(())
    }
}

impl Index<(usize, usize)> for Image {
    type Output = Radiance3;

    fn index<'a>(&'a self, idx: (usize, usize)) -> &'a Radiance3 {
        let (x, y) = idx;
        &self.data[x + y * self.width]
    }
}

impl IndexMut<(usize, usize)> for Image {
    fn index_mut<'a>(&'a mut self, idx: (usize, usize)) -> &'a mut Radiance3 {
        let (x, y) = idx;
        &mut self.data[x + y * self.width]
    }
}



fn visible(p: Vector3, direction: Vector3, dist: f32, scene: &Scene) -> bool {
    const RAY_BUMP_ELISION: f32 = 1e-4;

    let shadow_ray = Ray {
        origin: p + direction * RAY_BUMP_ELISION,
        direction: direction
    };

    let distance = dist - RAY_BUMP_ELISION;

    scene.triangles.iter().all(|triangle| {
        let (triangle_distance, _) = intersect(&shadow_ray, triangle);

        // The triangle must not be closer than the light
        triangle_distance >= distance
    })
}

fn shade(scene: &Scene, t: &Triangle, p: Point3, n: Vector3, w_o: Vector3) -> Radiance3 {
    let mut l_o = Vector3::zero();

    for light in &scene.lights {
        let offset = light.position - p;
        let distance = offset.length();
        let w_i = offset / distance;

        if visible(p, w_i, distance, scene) {
            let l_i = light.power / (4. * PI * distance.powi(2));

            l_o = l_o + l_i * w_i.dot(n).max(0.) * t.evaluate_finite_scattering_density(&w_i, &w_o).vec();
        }
    }

    Color3::from(l_o)
}

fn compute_eye_ray(x: f32, y: f32, width: usize, height: usize, camera: &Camera) -> Ray {
    let aspect = (height as f32) / (width as f32);

    let s = -2.0 * (camera.fov_x * 0.5).tan();

    let start = Vector3 {
        x:  ((x / (width as f32)) - 0.5) * s,
        y: -((y / (height as f32)) - 0.5) * s * aspect,
        z: 1.0
    } * camera.z_near;

    Ray {
        origin: start,
        direction: start.normalize(),
    }
}

fn intersect(r: &Ray, t: &Triangle) -> (f32, [f32; 3]) {
    let e1 = t.vertex[1] - t.vertex[0];
    let e2 = t.vertex[2] - t.vertex[0];
    let q = r.direction.cross(e2);

    let a = e1.dot(q);

    let s = r.origin - t.vertex[0];
    let sr = s.cross(e1);

    let mut weight = [0f32,
                  s.dot(q) / a,
                  r.direction.dot(sr) / a];
    weight[0] = 1. - (weight[1] + weight[2]);

    let dist = e2.dot(sr) / a;

    const EPSILON: f32 = 1e-7;
    const EPSILON2: f32 = 1e-10;

    (if (a <= EPSILON) || (dist <= 0.)
        || (weight[0] < -EPSILON2) || (weight[1] < -EPSILON2) || (weight[2] < -EPSILON2) {
        std::f32::INFINITY
    } else {
        dist
    }, weight)
}

fn sample_ray_triangle(scene: &Scene, _x: usize, _y: usize,
                       r: &Ray, t: &Triangle,
                       distance: &mut f32) -> Option<Radiance3> {
    let (d, weight) = intersect(r, t);
    if d >= *distance { return None; }
    *distance = d;

    let p = r.origin + r.direction * d;
    let n = t.normal.iter().zip(weight.iter()).map(|(&a, &b)| a * b)
        .fold(Vector3::zero(), |acc, x| acc + x).normalize();

    let w_o = -r.direction;

//    Some(Radiance3 { r: weight[0], g: weight[1], b: weight[2] } / 15f32)
    Some(shade(scene, t, p, n, w_o))
}

fn ray_trace(image: &mut Image, scene: &Scene, camera: &Camera,
             x0: usize, x1: usize, y0: usize, y1: usize) {
    for y in y0..y1 {
        for x in x0..x1 {
            let r = compute_eye_ray((x as f32) + 0.5, (y as f32) + 0.5,
                                    image.width(), image.height(), camera);

            let mut distance = std::f32::INFINITY;

            for t in &scene.triangles {
                //let Vector3 { x: vx, y: vy, z } = (r.direction + Vector3 { x: 1., y: 1., z: 1. }) / 5.;
                //image[(x, y)] = Color3 { r: vx, g: vy, b: z };
                if let Some(rad) = sample_ray_triangle(scene, x, y, &r, &t, &mut distance) {
                    image[(x, y)] = rad;
                }
            }
        }
    }
}

fn p3(x: f32, y: f32, z: f32) -> Point3 {
    Point3 { x: x, y: y, z: z }
}

fn test_scene() -> Scene {
    Scene {
        triangles: vec![Triangle {
            vertex: [p3(0., 1., -2.), p3(-1.9, -1., -2.), p3(1.6, -0.5, -2.)],
            normal: [p3( 0.0,  0.6,  1.0).normalize(),
                     p3(-0.4, -0.4,  1.0).normalize(),
                     p3( 0.4, -0.4,  1.0).normalize()],
            color: Color3 { r: 0., g: 0.8, b: 0. },
        }],
        lights: vec![Light {
            position: p3(1., 3., 1.),
            power: p3(10., 10., 10.),
        }],
    }
}

fn test_scene2() -> Scene {
    let mut scene = test_scene();

    scene.triangles.push(Triangle {
        vertex: [p3(-1.9, -1., -2.), p3(0., 1., -2.), p3(1.6, -0.5,-2.)],
        normal: [p3(-0.4, -0.4,  1.0).normalize(),
                 p3( 0.0,  0.6,  1.0).normalize(),
                 p3( 0.4, -0.4,  1.0).normalize()],
        color: Color3 { r: 0., g: 0.8, b: 0. },
    });

    const GROUND_Y: f32 = -1.;
    let ground_color = Color3 { r: 0.8, g: 0.8, b: 0.8 };
    let ny = p3(0., 1., 0.);
    scene.triangles.push(Triangle {
        vertex: [p3(-10., GROUND_Y, -10.), p3(-10., GROUND_Y, -0.01), p3(10., GROUND_Y, -0.01)],
        normal: [ny; 3],
        color: ground_color,
    });
    scene.triangles.push(Triangle {
        vertex: [p3(-1.9, -1., -2.), p3(0., 1., -2.), p3(1.6, -0.5,-2.)],
        normal: [ny; 3],
        color: ground_color,
    });    

    scene
}

fn main() {
    let scene = test_scene2();
    let camera = Camera::default();

    let sizex = 800;
    let sizey = 500;
    let mut img = Image::new(sizex, sizey);

    ray_trace(&mut img, &scene, &camera, 0, sizex, 0, sizey);
    
    img.save("out.ppm", 2.7).unwrap();
}

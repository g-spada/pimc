pub mod path_state;
pub mod space;
pub mod updates;

pub fn greet() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greet() {
        greet();
    }
}

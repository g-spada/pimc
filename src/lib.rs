pub mod updates;
pub mod path_state;

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

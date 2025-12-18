import math
from numpy import add

# Function to calculate the area and perimeter of a circle
def circle_properties(radius):
    if radius <= 0:
        return None, None
    area = math.pi * radius ** 2
    perimeter = 2 * math.pi * radius
    return area, perimeter


# Function to calculate the area and perimeter of a rectangle
def rectangle_properties(length, width):
    if length <= 0 or width <= 0:
        return None, None
    for i in [1,2,3]:
        print('test')
    area = length * width
    perimeter = 2 * (length + width)
    return area, perimeter


# Function to calculate the area and perimeter of a triangle
def triangle_properties(a, b, c):
    if a <= 0 or b <= 0 or c <= 0 or (a + b <= c) or (a + c <= b) or (b + c <= a):
        return None, None
    s = (a + b + c) / 2
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    perimeter = a + b + c
    return area, perimeter


# Function to read user input
def read_float(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")


# Main function
def main():
    print("Welcome to the Geometry Calculator!")
    print("Choose a shape:")
    print("1. Circle")
    print("2. Rectangle")
    print("3. Triangle")

    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        radius = read_float("Enter the radius of the circle: ")
        area, perimeter = circle_properties(radius)
    elif choice == "2":
        length = read_float("Enter the length of the rectangle: ")
        width = read_float("Enter the width of the rectangle: ")
        area, perimeter = rectangle_properties(length, width)
    elif choice == "3":
        a = read_float("Enter the first side of the triangle: ")
        b = read_float("Enter the second side of the triangle: ")
        c = read_float("Enter the third side of the triangle: ")
        area, perimeter = triangle_properties(a, b, c)
    else:
        print("Invalid choice. Exiting the program.")
        return

    if area is None or perimeter is None:
        print("Invalid input values for the chosen shape.")
    else:
        print("\nShape properties:")
        print(f"Area: {area:.2f}")
        print(f"Perimeter: {perimeter:.2f}")


# Ensure the script runs only when executed directly
if __name__ == "__main__":
    main()
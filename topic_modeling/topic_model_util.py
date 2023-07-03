import random
def take_dict_value(key, dictionary):
    try:
        return dictionary[key]
    except:
        return key

def generate_random_color():
    # Generate random RGB
    red = random.randint(10, 150)
    green = random.randint(50, 255)  # Adjusted range for vivid green
    blue = random.randint(50, 255)  # Adjusted range for vivid blue
    # Return the RGB color tuple
    return red, green, blue


# Getting a random red color for showing a negative topic
def generate_strong_red_color():
    # Generate random strong red RGB values
    red = random.randint(200, 255)  # Adjusted range for strong red
    green = random.randint(0, 50)
    blue = random.randint(0, 50)
    # Return the RGB color tuple
    return red, green, blue
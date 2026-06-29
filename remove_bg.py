from PIL import Image
import numpy as np

def remove_background(input_path, output_path):
    # Load image and convert to RGBA
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)
    
    # Calculate grayscale lightness
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # The logo is dark on a light background.
    # We can use the brightness to determine transparency.
    # The background is roughly RGB (243, 243, 243)
    # Let's find the base background color by taking the most common edge pixel or just a corner pixel
    bg_color = data[0,0][:3]
    
    # Calculate distance from background color
    dist = np.sqrt((r - bg_color[0])**2 + (g - bg_color[1])**2 + (b - bg_color[2])**2)
    
    # Map distance to alpha. If dist is 0 (exact background), alpha=0.
    # If dist is large (dark logo), alpha=255.
    # We'll use a threshold.
    max_dist = 50  # anything within 50 distance is considered background/fringe
    
    # Smooth blending for anti-aliasing
    alpha_mask = np.clip(dist * (255.0 / 100.0), 0, 255).astype(np.uint8)
    
    # Instead of distance, another way is to just use grayscale value:
    # background is white/light grey, logo is dark.
    # Alpha = 255 for black, 0 for white.
    grayscale = 0.299*r + 0.587*g + 0.114*b
    # Map grayscale to alpha: 
    # Background is ~240. Logo is ~50.
    # alpha = 255 * (240 - grayscale) / (240 - 50)
    alpha = np.clip(255 * (240 - grayscale) / (240 - 40), 0, 255).astype(np.uint8)
    
    # Set the alpha channel
    data[:,:,3] = alpha
    
    # For the colors, since it's anti-aliased on a light background, 
    # the fringe pixels might have light colors. Let's just make the whole logo purely dark grey.
    # The logo color is roughly (50, 50, 50).
    data[:,:,0] = 50
    data[:,:,1] = 50
    data[:,:,2] = 50
    
    out_img = Image.fromarray(data)
    out_img.save(output_path, "PNG")
    print(f"Background removed and saved to {output_path}")

if __name__ == "__main__":
    remove_background(r"d:\Project PHM\CorDeep\static\logo.png", r"d:\Project PHM\CorDeep\static\logo.png")

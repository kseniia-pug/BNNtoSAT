# The input on which the networks are not equal


pixels = s.split()
new_pixels = []
for pixel in pixels:
    if int(pixel) < 0:
        new_pixels.append('█')
    else:
        new_pixels.append(' ')

for i in range(len(new_pixels)):
    if i % 28 == 0:
        print('\n', end='')
    print(new_pixels[i], end='')
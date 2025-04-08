initial_size = (1280,720)
key_point = (230, 578)

scaled_size = (224, 224)

scale_factor = scaled_size[0] / initial_size[0]

scaled_key_point = (int(key_point[0] * scale_factor), int(key_point[1] * scale_factor))

print(scaled_key_point)

normalized_key_point = (scaled_key_point[0] / scaled_size[0], scaled_key_point[1] / scaled_size[1])

print(normalized_key_point)

restored_key_point = (int(normalized_key_point[0] * scaled_size[0]), int(normalized_key_point[1] * scaled_size[1]))

print(restored_key_point)
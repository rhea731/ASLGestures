import os

DATASET_PATH = "asl_alphabet_train"

classes = sorted(os.listdir(DATASET_PATH))

print("Classes:")
for class_name in classes:
    print(class_name)

print("\nNumber of classes:", len(classes))

for class_name in classes:
    class_path = os.path.join(DATASET_PATH, class_name)
    images = os.listdir(class_path)
    print(f"{class_name}: {len(images)} images")
import os


class VisDroneCaptioner:

    def __init__(self):

        # VisDrone Categories
        self.categories = {
            1: 'pedestrian',
            2: 'person',
            3: 'bicycle',
            4: 'car',
            5: 'van',
            6: 'truck',
            7: 'tricycle',
            8: 'awning-tricycle',
            9: 'bus',
            10: 'motor'
        }

    # ---------------------------------------------------
    # Generate Semantic Caption from Label File
    # ---------------------------------------------------

    def generate_caption(self, label_path):

        # Fallback caption
        if not os.path.exists(label_path):
            return "An aerial drone view of an urban scene."

        counts = {}

        with open(label_path, 'r') as f:

            for line in f.readlines():

                data = line.strip().split(',')

                # Safety check
                if len(data) >= 6:

                    try:
                        category_id = int(data[5])

                        if category_id in self.categories:

                            obj_name = self.categories[category_id]

                            counts[obj_name] = (
                                counts.get(obj_name, 0) + 1
                            )

                    except ValueError:
                        continue

        # No objects found
        if not counts:
            return (
                "An aerial drone view showing "
                "a clear urban scene."
            )

        # ---------------------------------------------------
        # Build Natural Caption
        # ---------------------------------------------------

        items = []

        for name, count in counts.items():

            if count > 1:
                items.append(f"{count} {name}s")
            else:
                items.append(f"{count} {name}")

        # Sentence formatting
        if len(items) == 1:

            caption = (
                f"An aerial drone view featuring "
                f"{items[0]}."
            )

        else:

            caption = (
                "An aerial drone view featuring "
                + ", ".join(items[:-1])
                + f", and {items[-1]}."
            )

        return caption


# ---------------------------------------------------
# QUICK TEST
# ---------------------------------------------------

if __name__ == "__main__":

    LABEL_DIR = (
        "data/VisDrone_Raw/VisDrone_Dataset/"
        "VisDrone2019-DET-train/labels"
    )

    sample_label = os.path.join(
        LABEL_DIR,
        os.listdir(LABEL_DIR)[0]
    )

    captioner = VisDroneCaptioner()

    caption = captioner.generate_caption(sample_label)

    print("\n--- VisDrone Caption Generator ---")

    print(f"Label File : {os.path.basename(sample_label)}")

    print(f"Caption    : {caption}")
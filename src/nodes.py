import json


class SplitKJCoordinatesLoop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "coordinates": ("STRING",),
                "overlap": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("first_half", "second_half")
    DESCRIPTION = "Split a list of looped coordinates into two halves with an overlap."
    FUNCTION = "split"

    CATEGORY = "MeshPhysics/SplitKJCoordinatesLoop"

    def split(self, coordinates, overlap):
        list_of_lists = json.loads(coordinates)
        mid_point = len(list_of_lists[0]) // 2

        first_half = [
            json.dumps(coord_list[:mid_point + overlap])
            for coord_list in list_of_lists
        ]
        second_half = [
            json.dumps(coord_list[mid_point:] + coord_list[:overlap])
            for coord_list in list_of_lists
        ]
        return (first_half, second_half)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SplitKJCoordinatesLoop": SplitKJCoordinatesLoop
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "SplitKJCoordinatesLoop": "SplitKJCoordinatesLoop"
}

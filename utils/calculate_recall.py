import os
import json
import argparse
import re
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Summarize all recall_* values for all scenes.")
    parser.add_argument("--result_path", required=True, help="Root directory containing scene folders.")
    args = parser.parse_args()

    result_path = args.result_path
    scene_list = sorted([
        d for d in os.listdir(result_path)
        if os.path.isdir(os.path.join(result_path, d))
    ])

    recall_dict = defaultdict(list)

    print("────────────────────────────────────────")
    print("Recall Summary")
    print("────────────────────────────────────────")

    for scene in scene_list:
        scene_name = scene.split("-")[0]
        summary_path = os.path.join(result_path, scene, "summary.json")

        if not os.path.exists(summary_path):
            print(f"{scene_name:<20} : summary.json not found")
            continue

        try:
            with open(summary_path, "r") as f:
                data = json.load(f)

            sparse_data = data.get("sparse", {})
            recall_keys = [k for k in sparse_data.keys() if k.startswith("recall_")]

            if len(recall_keys) == 0:
                print(f"{scene_name:<20} : no recall_* keys found")
                continue

            print(f"{scene_name:<20} :")
            for k in sorted(recall_keys):
                v = sparse_data.get(k)
                if v is not None:
                    recall_dict[k].append(v)
                    print(f"  - {k:<20} = {v:.4f}")
                else:
                    print(f"  - {k:<20} = None")

        except json.JSONDecodeError:
            print(f"{scene_name:<20} : Invalid JSON file")

    if recall_dict:
        print("────────────────────────────────────────")
        print("Average recall values")
        print("────────────────────────────────────────")
        for k in sorted(recall_dict.keys()):
            values = recall_dict[k]
            if len(values) > 0:
                avg_val = sum(values) / len(values)
                print(f"{k:<20} : {avg_val:.4f}")
    else:
        print("\nNo valid recall values found.")

if __name__ == "__main__":
    main()

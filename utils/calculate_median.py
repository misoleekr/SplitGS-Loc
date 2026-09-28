import os
import json
import argparse
import re

def main():
    parser = argparse.ArgumentParser(description="Summarize median_te and median_ae for all scenes.")
    parser.add_argument("--result_path", required=True, help="Root directory containing scene folders.")
    args = parser.parse_args()

    result_path = args.result_path
    scene_list = sorted([
        d for d in os.listdir(result_path)
        if os.path.isdir(os.path.join(result_path, d))
    ])

    te_values = []
    ae_values = []

    print("────────────────────────────────────────")
    print("Median TE/AE Summary")
    print("────────────────────────────────────────")

    for scene in scene_list:
        scene_name = scene.split("-")[0]
        summary_path = os.path.join(result_path, scene, "summary.json")

        if not os.path.exists(summary_path):
            print(f"{scene_name:<20}: summary.json not found")
            continue

        try:
            with open(summary_path, "r") as f:
                data = json.load(f)

            block = data.get("sparse", data)
            te = block.get("median_te")
            ae = block.get("median_ae")

            if (te is not None) and (ae is not None):
                te_values.append(te)
                ae_values.append(ae)
                print(f"{scene_name:<20}: {te:.2f}cm, {ae:.3f}deg")

        except json.JSONDecodeError:
            print(f"{scene_name:<20}: Invalid JSON file")

    if te_values and ae_values:
        avg_te = sum(te_values) / len(te_values)
        avg_ae = sum(ae_values) / len(ae_values)
        print("────────────────────────────────────────")
        text = "Average"
        print(f"{text:<20}: {avg_te:.2f}cm, {avg_ae:.3f}deg")
    else:
        print("\nNo valid median_te/median_ae values found.")

if __name__ == "__main__":
    main()

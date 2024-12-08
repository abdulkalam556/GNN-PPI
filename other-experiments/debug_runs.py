from datasets import load_from_disk

dataset = load_from_disk("/blue/cnt5410/shaik.abdulkalam/PPI/data/processed/")
print(dataset)
print(dataset["train"][0]["protein_1_graph"].keys())
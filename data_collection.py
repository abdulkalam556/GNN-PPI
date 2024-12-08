import os
import requests
import numpy as np

def create_folder_structure():
    """Create the folder structure for the project."""
    folders = [
        "../data/raw",
        "../data/processed",
        "../data/embeddings"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("Folder structure created successfully!")

def download_pdb_files(pdb_ids, output_dir="../data/raw/"):
    """
    Download PDB files using direct URLs and save them in the specified directory.
    Args:
        pdb_ids (list): List of PDB IDs to download.
        output_dir (str): Directory to save the downloaded PDB files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # PDB download URL template
    pdb_url_template = "https://files.rcsb.org/download/{}.pdb"

    # Track failed downloads
    failed_downloads = []

    for pdb_id in pdb_ids:
        try:
            # Construct the download URL
            pdb_url = pdb_url_template.format(pdb_id)

            # Attempt to download the file
            response = requests.get(pdb_url)
            if response.status_code == 200:
                # Save the file
                output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
                with open(output_path, "w") as pdb_file:
                    pdb_file.write(response.text)
                print(f"Successfully downloaded: {pdb_id}")
            else:
                print(f"Failed to download {pdb_id}: HTTP {response.status_code}")
                failed_downloads.append(pdb_id)

        except Exception as e:
            print(f"Error downloading {pdb_id}: {e}")
            failed_downloads.append(pdb_id)

    # Log failed downloads
    if failed_downloads:
        failed_file = "../data/failed_downloads.txt"
        with open(failed_file, "w") as f:
            f.write("\n".join(failed_downloads))
        print(f"Some downloads failed. See {failed_file} for details.")
    return failed_downloads

            
# def find_missing_pdbs(pdb_ids, raw_dir="../data/raw/"):
#     """
#     Identify missing PDB files that failed to download.
#     Args:
#         pdb_ids (list): List of original PDB IDs.
#         raw_dir (str): Directory containing downloaded PDB files.
#     Returns:
#         missing_pdbs (list): List of missing PDB IDs.
#     """
#     downloaded_files = [f[3:7].lower() for f in os.listdir(raw_dir) if f.endswith(".ent")]
#     missing_pdbs = [pdb_id for pdb_id in pdb_ids if pdb_id.lower() not in downloaded_files]
#     return missing_pdbs



if __name__ == "__main__":
    # Step 1: Create Folder Structure
    create_folder_structure()
    
    # Step 2: List of PDB IDs to download
    # Load the .npy file
    data = np.load('../data/human_dataset.npy', allow_pickle=True)

    # Extract PDB IDs (columns 2 and 5, assuming consistent format)
    pdb_ids_set = set(data[:, 2]).union(set(data[:, 5]))

    # Display the unique PDB IDs
    pdb_ids = [str(item) for item in pdb_ids_set]
    print(f"Total unique PDB IDs: {len(pdb_ids)}")
    
    # Step 3: Download PDB Files
    missing_pdbs = download_pdb_files(pdb_ids)
    print(f"Missing PDBs: {len(missing_pdbs)}")

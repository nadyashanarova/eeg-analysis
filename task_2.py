import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

import config_emotions as cfg
import data_loader

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    if not os.path.exists(cfg.INPUT_DIR):
        print(f"Input directory '{cfg.INPUT_DIR}' not found.")
        return
    
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)

    resolver = data_loader.SubjectFileResolver(cfg.INPUT_DIR)
    valid_pairs, missing = resolver.resolve_pairs()
    
    print(f"Found {len(valid_pairs)} valid subjects.")
    print(f"Missing pairs for: {missing}")

    if not valid_pairs:
        return

    extractor = data_loader.EmotionExtractor()
    all_epochs = []

    for sub_id, files in tqdm(valid_pairs.items(), desc="Processing Subjects"):
        sub_df = extractor.process_subject(sub_id, files)
        if sub_df is not None:
            all_epochs.append(sub_df)

    if not all_epochs:
        return

    full_df = pd.concat(all_epochs, ignore_index=True)

    melted_df = full_df.melt(
        id_vars=['time_bin', 'Stimulus', 'Subject', 'Gender'], 
        value_vars=cfg.EMOTION_COLS,
        var_name='Metric', 
        value_name='Value'
    )
    
    melted_df['Metric'] = melted_df['Metric'].str.replace('PM.', '').str.replace('.Scaled', '')

    unique_stimuli = melted_df['Stimulus'].unique()
    
    for stim in tqdm(unique_stimuli, desc="Plotting"):
        stim_data = melted_df[melted_df['Stimulus'] == stim]
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=stim_data, x='time_bin', y='Value', hue='Metric')
        plt.title(f"{stim}: Emotional Response (All Subjects)")
        plt.xlabel("Time (s)")
        plt.ylabel("Change from Baseline")
        plt.axvline(0, color='black', linestyle='--')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"{stim}_General.png"))
        plt.close()

        genders_present = stim_data['Gender'].unique()
        if len(genders_present) > 1:
            try:
                g = sns.FacetGrid(stim_data, col="Metric", hue="Gender", 
                                  col_wrap=3, height=3, aspect=1.3,
                                  palette={'Male': 'blue', 'Female': 'red', 'Unknown': 'grey'})
                g.map(sns.lineplot, "time_bin", "Value")
                g.add_legend()
                g.fig.suptitle(f"{stim}: Gender Comparison", y=1.02)
                
                for ax in g.axes.flat:
                    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
                    ax.axhline(0, color='k', linewidth=0.5)
                
                plt.savefig(os.path.join(cfg.OUTPUT_DIR, f"{stim}_Gender.png"))
                plt.close()
            except Exception as e:
                print(f"Error plotting gender for {stim}: {e}")

    print("Done.")

if __name__ == '__main__':
    main()
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    print("Generating Confusion Matrix...")
    # Load predictions
    df = pd.read_csv('rf_yield_predictions.csv')
    
    # Calculate mean actual yields per crop
    mean_yields = df.groupby('crop')['actual_yield_kg_ha'].transform('mean')
    
    # Create binary classes: 'Above Average' vs 'Below Average'
    df['Actual_Class'] = (df['actual_yield_kg_ha'] >= mean_yields).map({True: 'Above Average', False: 'Below Average'})
    df['Predicted_Class'] = (df['predicted_yield_kg_ha'] >= mean_yields).map({True: 'Above Average', False: 'Below Average'})
    
    # Generate confusion matrix
    labels = ['Above Average', 'Below Average']
    cm = confusion_matrix(df['Actual_Class'], df['Predicted_Class'], labels=labels)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Greens', ax=ax, colorbar=False)
                
    plt.title('Yield Prediction Confusion Matrix\n(Discretized Regression Outputs)', pad=15)
    plt.ylabel('Actual Yield Status', labelpad=10)
    plt.xlabel('Predicted Yield Status', labelpad=10)
    plt.tight_layout()
    
    # Save
    out_file = 'yield_confusion_matrix.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved successfully to {out_file}")

if __name__ == '__main__':
    main()


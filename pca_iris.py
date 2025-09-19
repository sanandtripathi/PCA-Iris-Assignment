import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

def run_iris_pca():
    """
    This function performs PCA on the Iris dataset and saves a visualization.
    """
    # 1. Load the Dataset
    # We can load the Iris dataset directly from scikit-learn, which is very convenient.
    iris = load_iris()
    # The data is in iris.data, and the target (species) is in iris.target.
    # The feature names are in iris.feature_names.
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    
    # Map target numbers to actual species names for better plotting
    target_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    df['species'] = df['target'].map(target_names)
    
    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
    print("\n")

    # 2. Prepare the Data
    # Separate features (X) from the target (y)
    features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    X = df[features].values
    y = df['species'].values
    
    # Scale the features. This is CRITICAL for PCA.
    # It standardizes the data so that features with larger values don't dominate the algorithm.
    X_scaled = StandardScaler().fit_transform(X)
    
    # 3. Perform PCA
    # We are reducing the 4 features down to 2 principal components.
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    # Create a new DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=['PC 1', 'PC 2'])
    
    # Add the species column back for visualization
    pca_df['species'] = y
    
    print("PCA completed. First 5 rows of the new PCA DataFrame:")
    print(pca_df.head())
    print("\n")

    # 4. Analyze the Results
    # 'explained_variance_ratio_' shows how much information (variance)
    # is captured by each principal component.
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by PC1: {explained_variance[0]:.2%}")
    print(f"Explained variance by PC2: {explained_variance[1]:.2%}")
    print(f"Total variance captured by both components: {explained_variance.sum():.2%}")
    print("\n")


    # 5. Visualize the PCA results
    # This plot is the key result for your report.
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC 1', y='PC 2', hue='species', data=pca_df, palette='viridis', s=100)
    
    plt.title('PCA of Iris Dataset', fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=15)
    plt.ylabel('Principal Component 2', fontsize=15)
    plt.legend()
    plt.grid()
    
    # Save the plot to a file
    plt.savefig('iris_pca_plot.png')
    print("Plot saved as 'iris_pca_plot.png'. You can now see the file.")
    
    # Show the plot
    plt.show()


# Run the main function
if __name__ == '__main__':
    run_iris_pca()

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configurações
COLIN_THRESHOLD = 0.6  # 60% para colinearidade
VIF_THRESHOLD = 5      # Limiar para multicolinearidade
INPUT_FILE = 'blind_test_sample.csv'
OUTPUT_TXT = 'resultados_completos.txt'
CORR_PLOT = 'matriz_correlacao.png'

def save_to_txt(content, file_path, mode='a'):
    """Salva conteúdo em arquivo TXT"""
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(content + "\n")

def load_and_process():
    try:
        with open(OUTPUT_TXT,'w') as f:
            f.write('RELATÓRIO DE ANÁLISE ESTATÍSTICA\n')
            f.write('='*50 +'\n\n')
        
        db = pd.read_csv(INPUT_FILE, encoding = 'utf-8').dropna()
        save_to_txt(f"Dados caregados: {db.shape[0]} registros, {db.shape[1]} colunas", OUTPUT_TXT)

        # Verifica se há variáveis categóricas
        categorical_cols = db.select_dtypes(include=['object','category']).columns
        if not categorical_cols.empty:
            save_to_txt(f'\nColunas categóricas detectadas: {list(categorical_cols)}',OUTPUT_TXT)
            # Tratamento das categorias raras
            for col in categorical_cols:
                db[col] = group_rare_categories(db[kcol])
            db = pd.get_dummies(db, drop_first=True)
            save_to_txt(f"\nApós one-hot encoding: {bdb.shape[1]} variáveis", OUTPUT_TXT)
        else:
            save_to_txt("\nNenhuma coluna categórica detectada.", OUTPUT_TXT)
        
        # Prepara X e y
        target_col = 'y_yes' if 'y_yes' in db.columns else 'y'
        if target_col not in db.columns:
            raise ValueError(f"Variável target '{target_col}' não encontrada no DataFrame.")
        y = db[target_col]
        X = db.drop(columns = [target_col])

        # Converter para numérico
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X = X.select_dtypes(include = [np.number])
        save_to_txt(f"\nVariáveis numéricas finais: {X.shape[1]}", OUTPUT_TXT)

        return X, y 
    except Exception as e:
        save_to_txt(f'\nERRO: {str(e)}',OUTPUT_TXT)
        exit()


def main():
    """Fluxo principal de execução"""
    print("Iniciando análise...")
    X, y = load_and_process() 


if __name__ == "__main__":
    main()
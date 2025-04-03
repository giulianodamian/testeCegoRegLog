import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configurações
PVALUE_THRESHOLD = 0.25  # Limiar de p-value para remoção inicial
INFLUENCE_THRESHOLD = 0.2  # Variação máxima permitida nos p-values (20%)
MAX_ITERATIONS = 100  # Máximo de iterações para ajuste
INPUT_FILE = 'blind_test_sample.csv'
OUTPUT_TXT = 'resultados_completos.txt'

def save_to_txt(content, file_path, mode='a'):
    """Salva conteúdo em arquivo TXT"""
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(content + "\n")

def group_rare_categories(series, threshold=0.05):
    """Agrupa categorias raras em uma única categoria 'Outros'"""
    counts = series.value_counts(normalize=True)
    rare_categories = counts[counts < threshold].index
    return series.replace(rare_categories, 'Outros')

def load_and_process():
    """Carrega e prepara os dados"""
    try:
        with open(OUTPUT_TXT, 'w') as f:
            f.write('RELATÓRIO DE SELEÇÃO DE VARIÁVEIS\n')
            f.write('='*50 + '\n\n')
        
        # Carrega os dados
        db = pd.read_csv(INPUT_FILE, encoding='utf-8').dropna()
        save_to_txt(f"Dados carregados: {db.shape[0]} registros, {db.shape[1]} colunas", OUTPUT_TXT)

        # Trata variáveis categóricas
        categorical_cols = db.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            save_to_txt(f'\nColunas categóricas detectadas: {list(categorical_cols)}', OUTPUT_TXT)
            for col in categorical_cols:
                db[col] = group_rare_categories(db[col])
            db = pd.get_dummies(db, drop_first=True)
            save_to_txt(f"\nApós one-hot encoding: {db.shape[1]} variáveis", OUTPUT_TXT)
        
        # Prepara X e y
        target_col = 'y_yes' if 'y_yes' in db.columns else 'y'
        if target_col not in db.columns:
            raise ValueError(f"Variável target '{target_col}' não encontrada.")
        
        y = db[target_col]
        X = db.drop(columns=[target_col])
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X = X.select_dtypes(include=[np.number])
        
        save_to_txt(f"\nVariáveis numéricas finais: {X.shape[1]}", OUTPUT_TXT)
        return X, y, X.columns.tolist()
    
    except Exception as e:
        save_to_txt(f'\nERRO: {str(e)}', OUTPUT_TXT)
        raise

def logistic_regression_pvalues(X, y):
    """Calcula p-values usando regressão logística"""
    X = sm.add_constant(X)  # Adiciona intercepto
    model = sm.Logit(y, X).fit(disp=0)
    return model.pvalues[1:]  # Remove p-value do intercepto

def select_features(X, y, feature_names=None):
    """Seleção iterativa de features baseada em p-values"""
    try:
        X_array = np.array(X)
        y_array = np.array(y).flatten()
        
        if feature_names is None and hasattr(X, 'columns'):
            feature_names = X.columns.tolist()
        
        # 1. Filtragem inicial por p-value
        pvalues = logistic_regression_pvalues(X_array, y_array)
        keep = [i for i, p in enumerate(pvalues) if p < PVALUE_THRESHOLD]
        removed = [i for i, p in enumerate(pvalues) if p >= PVALUE_THRESHOLD]
        
        original_pvalues = pvalues.copy()
        iteration = 0
        changed = True
        
        # 2. Ajuste iterativo
        while changed and iteration < MAX_ITERATIONS:
            iteration += 1
            changed = False
            
            if not keep:  # Se nenhuma feature foi mantida
                break
                
            current_pvalues = logistic_regression_pvalues(X_array[:, keep], y_array)
            
            # Verifica variações significativas
            for idx, kept_idx in enumerate(keep):
                original_p = original_pvalues[kept_idx]
                current_p = current_pvalues[idx]
                
                if original_p > 0:  # Evita divisão por zero
                    pct_change = abs((current_p - original_p) / original_p)
                    
                    if pct_change > INFLUENCE_THRESHOLD:
                        # Encontra melhor feature para readicionar
                        best_candidate = None
                        best_improvement = 0
                        
                        for candidate in removed:
                            # Testa readicionar temporariamente
                            temp_keep = keep + [candidate]
                            temp_pvalues = logistic_regression_pvalues(X_array[:, temp_keep], y_array)
                            
                            # Calcula melhoria
                            new_p = temp_pvalues[-1]  # P-value da feature readicionada
                            improvement = original_p - new_p  # Quanto reduziu o p-value
                            
                            if improvement > best_improvement:
                                best_improvement = improvement
                                best_candidate = candidate
                        
                        # Readiciona a melhor feature
                        if best_candidate is not None:
                            keep.append(best_candidate)
                            removed.remove(best_candidate)
                            changed = True
                            
                            name = feature_names[best_candidate] if feature_names else f"Feature {best_candidate}"
                            msg = f"Iteração {iteration}: Readicionada {name} (variação: {pct_change:.1%})"
                            print(msg)
                            save_to_txt(msg, OUTPUT_TXT)
                            break
            
            # Log do status
            if feature_names:
                current_names = [feature_names[i] for i in keep]
                status = f"Iteração {iteration}: {len(keep)} features ({', '.join(current_names)})"
            else:
                status = f"Iteração {iteration}: {len(keep)} features (índices: {keep})"
            
            print(status)
        
        # 3. Resultados finais
        X_filtered = X_array[:, keep]
        X_removed = X_array[:, removed]
        
        if feature_names:
            kept_names = [feature_names[i] for i in keep]
            removed_names = [feature_names[i] for i in removed]
        else:
            kept_names = keep
            removed_names = removed
        
        # Relatório final
        report = [
            "\n=== RESULTADO FINAL ===",
            f"Threshold de p-value: {PVALUE_THRESHOLD}",
            f"Threshold de influência: {INFLUENCE_THRESHOLD:.0%}",
            f"Total de iterações: {iteration}",
            f"\nFeatures mantidas ({len(keep)}):"
        ]
        
        for i in keep:
            name = feature_names[i] if feature_names else f"Feature {i}"
            original_p = original_pvalues[i]
            final_p = logistic_regression_pvalues(X_array[:, keep], y_array)[keep.index(i)]
            report.append(f"- {name}: p-value original={original_p:.4f}, final={final_p:.4f}")
        
        report.append(f"\nFeatures removidas ({len(removed)}):")
        for i in removed:
            name = feature_names[i] if feature_names else f"Feature {i}"
            report.append(f"- {name}: p-value={original_pvalues[i]:.4f}")
        
        final_report = "\n".join(report)
        print(final_report)
        save_to_txt(final_report, OUTPUT_TXT)
        
        return X_filtered, X_removed, kept_names, removed_names
    
    except Exception as e:
        error_msg = f"\nERRO: {str(e)}"
        print(error_msg)
        save_to_txt(error_msg, OUTPUT_TXT)
        raise

def main():
    """Fluxo principal"""
    print("Iniciando análise...")
    try:
        X, y, feature_names = load_and_process()
        select_features(X, y, feature_names)
    except Exception as e:
        print(f"Erro na execução: {str(e)}")
        save_to_txt(f"Erro na execução: {str(e)}", OUTPUT_TXT)

if __name__ == "__main__":
    main()
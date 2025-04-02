"""

ANÁLISE COMPLETA COM RELATÓRIO TXT E GRÁFICO DE CORRELAÇÃO
Gera:
1. Um arquivo TXT com todos os resultados
2. Gráfico da matriz de correlação
3. Tratamento sequencial de colinearidade e multicolinearidade
"""

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
import warnings
warnings.filterwarnings('ignore')

# Configurações
COLIN_THRESHOLD = 0.7  # 60% para colinearidade
VIF_THRESHOLD = 5      # Limiar para multicolinearidade
INPUT_FILE = 'blind_test_sample.csv'
OUTPUT_TXT = 'resultados_completos.txt'
CORR_PLOT = 'matriz_correlacao.png'

def save_to_txt(content, file_path, mode='a'):
    """Salva conteúdo em arquivo TXT"""
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(content + "\n")

def load_and_preprocess():
    """Carrega e prepara os dados"""
    try:
        # Inicializa arquivo TXT
        with open(OUTPUT_TXT, 'w') as f:
            f.write("RELATÓRIO DE ANÁLISE ESTATÍSTICA\n")
            f.write("="*50 + "\n\n")
        
        bank = pd.read_csv(INPUT_FILE, encoding='utf-8').dropna()
        save_to_txt(f"Dados carregados: {bank.shape[0]} registros, {bank.shape[1]} colunas", OUTPUT_TXT)
        
        # Tratamento de categorias raras
        def group_rare_categories(series, threshold=0.05):
            counts = series.value_counts(normalize=True)
            rare_categories = counts[counts < threshold].index
            return series.replace(rare_categories, 'Other')
        
        categorical_cols = bank.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            bank[col] = group_rare_categories(bank[col])
        
        bank = pd.get_dummies(bank, drop_first=True)
        save_to_txt(f"\nApós one-hot encoding: {bank.shape[1]} variáveis", OUTPUT_TXT)
        
        # Preparar X e y
        target_col = 'y_yes' if 'y_yes' in bank.columns else 'y'
        y = bank[target_col]
        X = bank.drop(columns=[target_col])
        
        # Converter para numérico
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X = X.select_dtypes(include=[np.number])
        save_to_txt(f"\nVariáveis numéricas finais: {X.shape[1]}", OUTPUT_TXT)
        
        return X, y
    except Exception as e:
        save_to_txt(f"\nERRO: {str(e)}", OUTPUT_TXT)
        exit()

def plot_correlation_matrix(X, threshold):
    """Gera e salva gráfico da matriz de correlação completa"""
    plt.figure(figsize=(15, 12))
    corr_matrix = X.corr().abs()
    
    # Cria máscara para esconder a diagonal principal (opcional)
    mask = np.zeros_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, True)  # Se quiser esconder a diagonal
    
    sns.heatmap(corr_matrix, 
                mask=mask,  # Remova esta linha para mostrar tudo
                annot=True, 
                fmt=".2f", 
                cmap='coolwarm',
                annot_kws={"size": 8}, 
                vmin=-1, 
                vmax=1,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    
    plt.title(f'Matriz de Correlação Completa (|r| ≥ {threshold})', 
              pad=20, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adiciona linhas para destacar a diagonal principal
    for i in range(len(corr_matrix)):
        plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='black', lw=1))
    
    plt.tight_layout()
    plt.savefig(CORR_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    save_to_txt(f"\nMatriz de correlação completa salva em: {CORR_PLOT}", OUTPUT_TXT)
    
def remove_highly_correlated(X, threshold):
    """Remove variáveis com correlação acima do threshold"""
    save_to_txt(f"\nTRATAMENTO DE COLINEARIDADE (≥{threshold*100}%)", OUTPUT_TXT)
    
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    # Relatório
    report = []
    for col in to_drop:
        corr_with = upper[col].idxmax()
        corr_value = corr_matrix.loc[col, corr_with]
        report.append({
            'Variável': col,
            'Correlacionada com': corr_with,
            'Correlação': f"{corr_value:.2f}"
        })
    
    X_filtered = X.drop(columns=to_drop, errors='ignore')
    
    # Salva relatório
    save_to_txt(f"\nVariáveis removidas: {len(to_drop)}", OUTPUT_TXT)
    if report:
        df_report = pd.DataFrame(report)
        save_to_txt(df_report.to_string(index=False), OUTPUT_TXT)
    
    return X_filtered, pd.DataFrame(report)

def remove_high_vif(X, threshold):
    """Remove variáveis com alto VIF"""
    save_to_txt(f"\nTRATAMENTO DE MULTICOLINEARIDADE (VIF > {threshold})", OUTPUT_TXT)
    
    X_clean = X.copy()
    vif_reports = []
    
    for i in range(X.shape[1]):  # Número máximo de iterações
        if X_clean.shape[1] <= 1:  # Garante que pelo menos 1 variável permaneça
            break
            
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_clean.columns
        try:
            vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i) 
                              for i in range(X_clean.shape[1])]
        except:
            break
            
        vif_data = vif_data.sort_values("VIF", ascending=False)
        max_vif = vif_data["VIF"].max()
        
        if max_vif <= threshold:
            break
            
        var_to_remove = vif_data.iloc[0]["Variable"]
        X_clean = X_clean.drop(columns=[var_to_remove], errors='ignore')
        
        # Relatório
        vif_reports.append({
            'Iteração': i+1,
            'Variável Removida': var_to_remove,
            'VIF': f"{max_vif:.2f}",
            'Variáveis Restantes': X_clean.shape[1]
        })
    
    # Salva relatório
    save_to_txt(f"\nVariáveis removidas: {len(vif_reports)}", OUTPUT_TXT)
    if vif_reports:
        df_report = pd.DataFrame(vif_reports)
        save_to_txt(df_report.to_string(index=False), OUTPUT_TXT)
    
    return X_clean, pd.DataFrame(vif_reports)

def run_final_model(X, y):
    """Executa a modelagem final"""
    save_to_txt("\nMODELAGEM FINAL", OUTPUT_TXT)
    
    # Padronização
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Seleção de features
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=StratifiedKFold(5),
        scoring='accuracy',
        min_features_to_select=1
    )
    rfecv.fit(X_scaled, y)
    
    # Modelo final
    X_final = sm.add_constant(X_scaled[X_scaled.columns[rfecv.support_]])
    logit_model = sm.Logit(y, X_final)
    result = logit_model.fit()
    
    # Salva sumário
    save_to_txt("\nRESUMO DO MODELO FINAL:", OUTPUT_TXT)
    save_to_txt(result.summary().as_text(), OUTPUT_TXT)
    
    # Ranking de features
    ranking_df = pd.DataFrame({
        'Feature': X_scaled.columns,
        'Ranking': rfecv.ranking_,
        'Suporte': rfecv.support_
    }).sort_values('Ranking')
    
    save_to_txt("\nRANKING DE VARIÁVEIS:", OUTPUT_TXT)
    save_to_txt(ranking_df.to_string(index=False), OUTPUT_TXT)
    
    # Coeficientes finais
    coef_table = result.summary2().tables[1]
    coef_table['abs_z'] = np.abs(coef_table['z'])
    significant_vars = coef_table[coef_table['P>|z|'] < 0.05].sort_values('abs_z', ascending=False)
    
    save_to_txt("\nVARIÁVEIS SIGNIFICATIVAS (p < 0.05):", OUTPUT_TXT)
    save_to_txt(significant_vars[['Coef.', 'P>|z|']].to_string(), OUTPUT_TXT)
    
    return result, rfecv, ranking_df

def main():
    """Fluxo principal de execução"""
    print("Iniciando análise...")
    
    # 1. Carregar e preparar dados
    X, y = load_and_preprocess()
    
    # 2. Matriz de correlação
    plot_correlation_matrix(X, COLIN_THRESHOLD)
    
    # 3. Tratamento de colinearidade
    X_colin, colin_report = remove_highly_correlated(X, COLIN_THRESHOLD)
    save_to_txt(f"\nVariáveis após colinearidade: {X_colin.shape[1]}", OUTPUT_TXT)
    
    # 4. Tratamento de multicolinearidade
    X_vif, vif_report = remove_high_vif(X_colin, VIF_THRESHOLD)
    save_to_txt(f"\nVariáveis após multicolinearidade: {X_vif.shape[1]}", OUTPUT_TXT)
    
    # 5. Modelagem final
    result, rfecv, ranking_df = run_final_model(X_vif, y)
    save_to_txt(f"\nVariáveis selecionadas: {sum(rfecv.support_)}", OUTPUT_TXT)
    
    # 6. Salva resultados adicionais
    ranking_df.to_csv('ranking_features.csv', index=False)
    pd.DataFrame({'features_selecionadas': X_vif.columns[rfecv.support_]}).to_csv('features_selecionadas.csv', index=False)
    
    print("\nAnálise concluída com sucesso!")
    print(f"Resultados salvos em: {OUTPUT_TXT} e {CORR_PLOT}")

if __name__ == "__main__":
    main()
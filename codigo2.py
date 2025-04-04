import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2
from patsy import dmatrix
from scipy.special import expit
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

# Configurações
PVALUE_THRESHOLD = 0.25
INFLUENCE_THRESHOLD = 0.2
MAX_ITERATIONS = 1000
INPUT_FILE = 'blind_test_sample.csv'
OUTPUT_TXT = 'resultados_completos.txt'
OUTPUT_PNG_PREFIX = 'scatter_plot_'
NONLINEAR_THRESHOLD = 0.1
HOSMER_LEMESHOW_GROUPS = 10
COLIN_THRESHOLD = 0.6
VIF_THRESHOLD = 5
CORR_PLOT = 'matriz_correlacao.png'

def save_to_txt(content, file_path, mode='a'):
    """Salva conteúdo em arquivo TXT."""
    with open(file_path, mode, encoding='utf-8') as f:
        f.write(content + "\n")

def load_and_process():
    """Carrega e prepara os dados."""
    try:
        with open(OUTPUT_TXT, 'w') as f:
            f.write('RELATÓRIO DE ANÁLISE ESTATÍSTICA\n')
            f.write('=' * 50 + '\n\n')

        # Carrega e verifica os dados
        db = pd.read_csv(INPUT_FILE, encoding='utf-8')
        if db.empty:
            raise ValueError("Arquivo vazio ou sem dados válidos.")

        db = db.dropna()
        if db.empty:
            raise ValueError("Todos os dados removidos durante limpeza.")

        save_to_txt(f"Dados carregados: {db.shape[0]} registros, {db.shape[1]} colunas", OUTPUT_TXT)

        # Prepara variáveis
        target_col = 'y_yes' if 'y_yes' in db.columns else 'y'
        if target_col not in db.columns:
            raise ValueError(f"Variável target '{target_col}' não encontrada.")

        y = db[target_col]
        X = db.drop(columns=[target_col])

        # Processa variáveis numéricas
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        X = X.select_dtypes(include=[np.number])

        if X.empty:
            raise ValueError("Nenhuma variável numérica válida encontrada.")

        return X, y, X.columns.tolist()

    except Exception as e:
        save_to_txt(f'\nERRO: {str(e)}', OUTPUT_TXT)
        raise

def logistic_regression_pvalues(X, y):
    """Calcula p-values usando regressão logística."""
    try:
        X = sm.add_constant(X)
        model = sm.Logit(y, X).fit(disp=0)
        return model.pvalues[1:]
    except Exception as e:
        print(f"Erro na regressão logística: {str(e)}")
        return np.array([])

def run_univariable_analysis(X, y, feature_names, output_txt):
    """Executa análise univariada."""

    save_to_txt("\n=== ANÁLISE UNIVARIADA ===", output_txt)
    univariable_results = []

    for feature in feature_names:
        X_uni = sm.add_constant(X[[feature]])
        model = sm.Logit(y, X_uni).fit(disp=0)
        p_value = model.pvalues[1]  # P-value da variável
        coef = model.params[1]
        std_err = model.bse[1]

        univariable_results.append({
            "Variable": feature,
            "Coefficient": coef,
            "Standard Error": std_err,
            "P-value": p_value
        })

        save_to_txt(f"\n--- {feature} ---", output_txt)
        save_to_txt(model.summary().as_text(), output_txt)

    df_results = pd.DataFrame(univariable_results)
    save_to_txt("\nResultados Univariados:", output_txt)
    save_to_txt(df_results.to_string(index=False), output_txt)

    # Seleciona variáveis para a próxima etapa (p < 0.25)
    selected_features = df_results[df_results['P-value'] < PVALUE_THRESHOLD]['Variable'].tolist()
    save_to_txt(f"\nVariáveis selecionadas para multivariada: {selected_features}", output_txt)

    return selected_features

def multivariable_model_comparison(X, y, initial_features, output_txt):
    """Compara modelos multivariados."""

    save_to_txt("\n=== COMPARAÇÃO DE MODELOS MULTIVARIADOS ===", output_txt)

    X_multi = sm.add_constant(X[initial_features])
    model_full = sm.Logit(y, X_multi).fit(disp=0)

    current_features = initial_features.copy()
    removed_features = []
    iteration = 0
    changed = True

    while changed and iteration < MAX_ITERATIONS:
        iteration += 1
        changed = False

        save_to_txt(f"\n--- Iteration {iteration} ---", output_txt)
        save_to_txt(f"Features no modelo: {current_features}", output_txt)

        # Encontra a variável com o maior p-value
        pvalues = logistic_regression_pvalues(X_multi[current_features], y)
        max_p_idx = np.argmax(pvalues)
        max_p_feature = current_features[max_p_idx]
        max_p_value = pvalues[max_p_idx]

        save_to_txt(f"Maior P-value: {max_p_feature} = {max_p_value:.4f}", output_txt)

        if max_p_value > 0.05:  # Usando 0.05 como limiar de significância
            removed_features.append(max_p_feature)
            current_features.remove(max_p_feature)
            X_reduced = sm.add_constant(X[current_features])
            model_reduced = sm.Logit(y, X_reduced).fit(disp=0)

            # Teste de razão de verossimilhança
            lr_test = sm.stats.anova_lr(model_reduced, model_full, typ=' Chisq')
            lr_pvalue = lr_test.values[1]
            save_to_txt(f"Teste de Razão de Verossimilhança P-value: {lr_pvalue:.4f}", output_txt)

            if lr_pvalue > 0.05:  # Não há diferença significativa
                save_to_txt(f"Removendo {max_p_feature}", output_txt)
                model_full = model_reduced
                X_multi = X_reduced
                changed = True

                # Verifica mudança nos coeficientes
                coef_full = model_full.params
                coef_reduced = model_reduced.params
                coef_change = {}
                for var in current_features:
                    if var in coef_full and var in coef_reduced:
                        coef_change[var] = abs((coef_reduced[var] - coef_full[var]) / coef_full[var]) if coef_full[var] != 0 else np.inf
                save_to_txt(f"Mudança nos coeficientes: {coef_change}", output_txt)

                if any(change > 0.2 for change in coef_change.values()):  # Se alguma mudança > 20%
                    save_to_txt(f"Re-adicionando {max_p_feature} devido a grande mudança nos coeficientes.", output_txt)
                    current_features.append(max_p_feature)
                    model_full = sm.Logit(y, sm.add_constant(X[current_features])).fit(disp=0)
                    changed = True  # Refaz a iteração
            else:
                save_to_txt(f"Mantendo {max_p_feature} devido a diferença significativa entre modelos.", output_txt)
        else:
            save_to_txt("Nenhuma variável removida.", output_txt)
            break  # Sai do loop se nenhum p-value for maior que 0.05

    save_to_txt("\n=== Modelo Final Multivariado ===", output_txt)
    save_to_txt(model_full.summary().as_text(), output_txt)

    return model_full, current_features, removed_features

def check_nonlinearity(x, y):
    """Verifica não-linearidade."""
    try:
        X_linear = sm.add_constant(x)
        model_linear = sm.OLS(y, X_linear).fit()

        X_quad = sm.add_constant(np.column_stack([x, x**2]))
        model_quad = sm.OLS(y, X_quad).fit()

        lr_stat = -2 * (model_linear.llf - model_quad.llf)
        return chi2.sf(lr_stat, df=1) < NONLINEAR_THRESHOLD, chi2.sf(lr_stat, df=1)
    except:
        return False, 1.0

def plot_scatter_smooth(df, feature_names, target_col, output_txt):
    """Gera gráficos de dispersão e detecta não-linearidade."""
    nonlinear_features = {}

    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        sns.regplot(x=feature, y=target_col, data=df, lowess=True,
                    scatter_kws={'s': 10}, line_kws={'color': 'red'})
        plt.title(f'Scatter Plot of {feature} vs. {target_col}')
        plt.xlabel(feature)
        plt.ylabel(target_col)

        is_nonlinear, p_value = check_nonlinearity(df[feature], df['target']) # Usando 'target' aqui
        plt.text(0.5, 0.9, f"{'Não-linear' if is_nonlinear else 'Linear'} (p={p_value:.4f})",
                transform=plt.gca().transAxes, color='red' if is_nonlinear else 'green')

        if is_nonlinear:
            nonlinear_features[feature] = p_value

        filename = f"{OUTPUT_PNG_PREFIX}{feature}.png"
        plt.savefig(filename)
        plt.close()

        save_to_txt(f"Scatter plot for {feature} saved to {filename}", output_txt)

    return nonlinear_features

def hosmer_lemeshow_test(y_true, y_prob, groups=10):
    """Calcula o teste de Hosmer-Lemeshow."""
    q_y = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    q_y['cut_y_prob'] = pd.qcut(q_y['y_prob'], groups)

    level_y = q_y.groupby('cut_y_prob').mean()
    level_y['total'] = q_y.groupby('cut_y_prob').size()

    hosmer_stat = np.sum((level_y['y_true'] - level_y['y_prob'])**2 /
                       (level_y['y_prob'] * (1 - level_y['y_prob']) / level_y['total']))

    dof = groups - 2
    p_value = chi2.sf(hosmer_stat, dof)
    return hosmer_stat, p_value

def calculate_deviance(y_true, y_pred):
    """Calcula a deviance para um modelo logístico."""
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)  # Evita valores extremos
    return -2 * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def calculate_pearson_chi2(y_true, y_pred):
    """Calcula a estatística de Pearson Chi-square."""
    residuals = y_true - y_pred
    pearson_residuals = residuals / np.sqrt(y_pred * (1 - y_pred))
    return np.sum(pearson_residuals**2)

def plot_correlation_matrix(X, threshold, output_txt, corr_plot_name):
    """Gera matriz de correlação mostrando todos os pontos, mas destacando os significativos"""
    plt.figure(figsize=(15, 12))

    # Calcula matriz de correlação
    corr_matrix = X.corr().abs()

    # Cria máscara para a diagonal principal
    mask = np.zeros_like(corr_matrix, dtype=bool)
    np.fill_diagonal(mask, True)

    # Plotagem da matriz completa
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 8})

    # Destacar correlações acima do threshold
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j and abs(corr_matrix.iloc[i, j]) >= threshold:
                plt.text(j + 0.5, i + 0.5,
                         f"{corr_matrix.iloc[i, j]:.2f}",
                         ha="center", va="center",
                         color="black", fontsize=10,
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.title(f'Matriz de Correlação (Destaque para |r| ≥ {threshold})', pad=20, fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(corr_plot_name, dpi=300, bbox_inches='tight')
    plt.close()

    save_to_txt(f"\nMatriz de correlação salva em: {corr_plot_name}", output_txt)

def remove_highly_correlated(X, threshold, output_txt):
    """Remove variáveis com correlação acima do threshold"""
    save_to_txt(f"\nTRATAMENTO DE COLINEARIDADE (≥{threshold*100}%)", output_txt)

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
    save_to_txt(f"\nVariáveis removidas: {len(to_drop)}", output_txt)
    if report:
        df_report = pd.DataFrame(report)
        save_to_txt(df_report.to_string(index=False), output_txt)

    return X_filtered, pd.DataFrame(report)

def remove_high_vif(X, threshold, output_txt):
    """Remove variáveis com alto VIF"""
    save_to_txt(f"\nTRATAMENTO DE MULTICOLINEARIDADE (VIF > {threshold})", output_txt)

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
            'Iteração': i + 1,
            'Variável Removida': var_to_remove,
            'VIF': f"{max_vif:.2f}",
            'Variáveis Restantes': X_clean.shape[1]
        })

    # Salva relatório
    save_to_txt(f"\nVariáveis removidas: {len(vif_reports)}", output_txt)
    if vif_reports:
        df_report = pd.DataFrame(vif_reports)
        save_to_txt(df_report.to_string(index=False), output_txt)

    return X_clean, pd.DataFrame(vif_reports)

def run_final_model(X, y, output_txt):
    """Executa a modelagem final"""
    save_to_txt("\nMODELAGEM FINAL", output_txt)

    # Padronização
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Seleção de features (RFE)
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
    save_to_txt("\nRESUMO DO MODELO FINAL:", output_txt)
    save_to_txt(result.summary().as_text(), output_txt)

    # Ranking de features
    ranking_df = pd.DataFrame({
        'Feature': X_scaled.columns,
        'Ranking': rfecv.ranking_,
        'Suporte': rfecv.support_
    }).sort_values('Ranking')

    save_to_txt("\nRANKING DE VARIÁVEIS:", output_txt)
    save_to_txt(ranking_df.to_string(index=False), output_txt)

    # Coeficientes finais
    coef_table = result.summary2().tables[1]
    coef_table['abs_z'] = np.abs(coef_table['z'])
    significant_vars = coef_table[coef_table['P>|z|'] < 0.05].sort_values('abs_z', ascending=False)

    save_to_txt("\nVARIÁVEIS SIGNIFICATIVAS (p < 0.05):", output_txt)
    save_to_txt(significant_vars[['Coef.', 'P>|z|']].to_string(), output_txt)

    return result, rfecv, ranking_df

def main():
    """Fluxo principal de execução"""
    print("Iniciando análise...")
    try:
        # 1. Carregar e preparar dados
        X, y, feature_names = load_and_process()

        # 2. Análise Univariada (Conforme Artigo)
        univariable_selected_features = run_univariable_analysis(X, y, feature_names, OUTPUT_TXT)
        
        # Verifica se há features selecionadas
        if not univariable_selected_features:
            raise ValueError("Nenhuma variável selecionada na análise univariada.")

        # 3. Verifica não-linearidade
        df = X.copy()
        df['target'] = y
        nonlinear_features = plot_scatter_smooth(df, feature_names, 'target', OUTPUT_TXT)
        
        if nonlinear_features:
            print(f"\nFeatures não-lineares detectadas: {list(nonlinear_features.keys())}")

        # 4. Tratamento de colinearidade
        plot_correlation_matrix(X, COLIN_THRESHOLD, OUTPUT_TXT, CORR_PLOT)
        X_colin, colin_report = remove_highly_correlated(X, COLIN_THRESHOLD, OUTPUT_TXT)
        save_to_txt(f"\nVariáveis após colinearidade: {X_colin.shape[1]}", OUTPUT_TXT)

        # 5. Tratamento de multicolinearidade
        X_vif, vif_report = remove_high_vif(X_colin, VIF_THRESHOLD, OUTPUT_TXT)
        save_to_txt(f"\nVariáveis após multicolinearidade: {X_vif.shape[1]}", OUTPUT_TXT)

        # Verifica quais features selecionadas ainda estão disponíveis
        available_features = [f for f in univariable_selected_features if f in X_vif.columns]
        
        if not available_features:
            raise ValueError("Nenhuma das variáveis selecionadas está disponível após tratamento de colinearidade.")
            
        save_to_txt(f"\nVariáveis disponíveis para modelagem multivariada: {available_features}", OUTPUT_TXT)

        # 6. Modelagem Multivariada (Conforme Artigo)
        final_model, final_features, removed_features = multivariable_model_comparison(
            X_vif, y, available_features, OUTPUT_TXT
        )

        # 7. Avaliação da Qualidade do Ajuste (GOF)
        if final_model is not None:
            X_final = sm.add_constant(X_vif[final_features])
            y_pred = final_model.predict(X_final)
            
            print("\nAnálise da Qualidade do Ajuste:")
            
            # Deviance (calculada manualmente)
            deviance = calculate_deviance(y, y_pred)
            print(f"Deviance: {deviance:.4f}")
            
            # Pearson Chi-square (calculado manualmente)
            pearson_chi2 = calculate_pearson_chi2(y, y_pred)
            print(f"Pearson Chi-square: {pearson_chi2:.4f}")
            
            # Hosmer-Lemeshow Test
            hl_stat, hl_p_value = hosmer_lemeshow_test(y, y_pred, groups=HOSMER_LEMESHOW_GROUPS)
            print(f"Hosmer-Lemeshow Chi2: {hl_stat:.4f}, p-value: {hl_p_value:.4f}")
            
            # Outras métricas
            print(f"\nOutras Métricas:")
            print(f"Log-Likelihood: {final_model.llf:.4f}")
            print(f"LL-Null: {final_model.llnull:.4f}")
            print(f"Pseudo R-squared: {final_model.prsquared:.4f}")

        # 8. Modelagem Final com RFECV
        final_result, rfecv, ranking_df = run_final_model(X_vif, y, OUTPUT_TXT)
        
        print("\nAnálise concluída com sucesso!")
        print(f"Resultados salvos em: {OUTPUT_TXT} e {CORR_PLOT}")

    except Exception as e:
        print(f"\nERRO: {str(e)}")
        save_to_txt(f"\nERRO: {str(e)}", OUTPUT_TXT)

if __name__ == "__main__":
    main()
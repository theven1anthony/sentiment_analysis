#!/usr/bin/env python3
"""
Script pour analyser et comparer les résultats des expérimentations MLflow.
"""

import mlflow
import pandas as pd
import click
from mlflow.tracking import MlflowClient

@click.command()
@click.option('--experiment-name',
              default='simple_models_v4',
              help='Nom de l\'expérimentation MLflow à analyser')
def main(experiment_name):
    """Analyse les résultats d'une expérimentation MLflow avec option CLI."""
    analyze_experiment(experiment_name)

def analyze_experiment(experiment_name="simple_models_v4"):
    """
    Analyse les résultats d'une expérimentation MLflow.
    """
    client = MlflowClient()

    # Récupérer l'expérimentation
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"Expérimentation '{experiment_name}' non trouvée")
            return None
    except Exception as e:
        print(f"Erreur: {e}")
        return None

    # Récupérer tous les runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"]
    )

    if not runs:
        print(f"Aucun run trouvé dans l'expérimentation '{experiment_name}'")
        return None

    # Extraire les données
    results = []
    for run in runs:
        run_data = {
            'run_id': run.info.run_id,
            'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
            'description': run.data.tags.get('description', 'N/A'),
            'technique': run.data.params.get('technique', 'N/A'),
            'handle_negations': run.data.params.get('handle_negations', 'N/A'),
            'handle_emotions': run.data.params.get('handle_emotions', 'N/A'),
            'accuracy': run.data.metrics.get('accuracy', 0),
            'f1_score': run.data.metrics.get('f1_score', 0),
            'precision': run.data.metrics.get('precision', 0),
            'recall': run.data.metrics.get('recall', 0),
            'auc_score': run.data.metrics.get('auc_score', 0),
            'training_time': run.data.metrics.get('training_time', 0),
            'start_time': run.info.start_time
        }
        results.append(run_data)

    # Créer DataFrame
    df = pd.DataFrame(results)

    # Afficher les résultats
    print(f"\n=== ANALYSE EXPÉRIMENTATION '{experiment_name}' ===")
    print(f"Nombre total de runs: {len(df)}")

    # Filtrer les runs de modèles (pas les summaries)
    model_runs = df[df['technique'].isin(['stemming', 'lemmatization'])].copy()

    if len(model_runs) == 0:
        print("Aucun run de modèle trouvé")
        return df

    print(f"Runs de modèles: {len(model_runs)}")

    # Analyser par configuration
    print("\n=== RÉSULTATS PAR CONFIGURATION ===")

    # Grouper par paramètres négations/émotions
    for (neg, emo), group in model_runs.groupby(['handle_negations', 'handle_emotions']):
        print(f"\n📋 Configuration: Négations={neg}, Émotions={emo}")
        print(f"Description: {group['description'].iloc[0]}")

        # Trouver la meilleure technique
        best_run = group.loc[group['f1_score'].idxmax()]

        print(f"🏆 Meilleure technique: {best_run['technique'].upper()}")
        print(f"   - F1-Score: {best_run['f1_score']:.4f}")
        print(f"   - Accuracy: {best_run['accuracy']:.4f}")
        if best_run['auc_score'] > 0:
            print(f"   - AUC-ROC: {best_run['auc_score']:.4f}")
        print(f"   - Training Time: {best_run['training_time']:.3f}s")

        # Afficher toutes les techniques pour cette config
        print("   Détail des techniques:")
        for _, row in group.iterrows():
            print(f"     {row['technique'].capitalize()}: F1={row['f1_score']:.4f}, Acc={row['accuracy']:.4f}, Time={row['training_time']:.3f}s")

    # Analyse comparative globale
    print("\n=== ANALYSE COMPARATIVE GLOBALE ===")

    # Tableau récapitulatif
    comparison_data = []
    for (neg, emo), group in model_runs.groupby(['handle_negations', 'handle_emotions']):
        best_run = group.loc[group['f1_score'].idxmax()]
        row = {
            'Configuration': f"Neg={neg}, Emo={emo}",
            'Meilleure_Technique': best_run['technique'].capitalize(),
            'F1_Score': best_run['f1_score'],
            'Accuracy': best_run['accuracy'],
            'Training_Time': best_run['training_time']
        }
        # Ajouter AUC si disponible
        if best_run['auc_score'] > 0:
            row['AUC_Score'] = best_run['auc_score']
        comparison_data.append(row)

    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('F1_Score', ascending=False)

    print("\n📊 Classement par performance (F1-Score):")
    print(df_comparison.to_string(index=False, float_format='%.4f'))

    # Analyse des tendances
    print("\n=== ANALYSE DES TENDANCES ===")

    # Impact des négations
    neg_true = model_runs[model_runs['handle_negations'] == 'True']['f1_score'].mean()
    neg_false = model_runs[model_runs['handle_negations'] == 'False']['f1_score'].mean()
    print(f"Impact des négations:")
    print(f"  - Avec négations: F1 moyen = {neg_true:.4f}")
    print(f"  - Sans négations: F1 moyen = {neg_false:.4f}")
    print(f"  - Différence: {neg_true - neg_false:+.4f}")

    # Impact des émotions
    emo_true = model_runs[model_runs['handle_emotions'] == 'True']['f1_score'].mean()
    emo_false = model_runs[model_runs['handle_emotions'] == 'False']['f1_score'].mean()
    print(f"\nImpact des émotions:")
    print(f"  - Avec émotions: F1 moyen = {emo_true:.4f}")
    print(f"  - Sans émotions: F1 moyen = {emo_false:.4f}")
    print(f"  - Différence: {emo_true - emo_false:+.4f}")

    # Technique préférée
    technique_performance = model_runs.groupby('technique')['f1_score'].mean()
    print(f"\nPerformance par technique:")
    for technique, score in technique_performance.items():
        print(f"  - {technique.capitalize()}: F1 moyen = {score:.4f}")

    # Analyse spécifique AUC-ROC
    auc_runs = model_runs[model_runs['auc_score'] > 0]
    if len(auc_runs) > 0:
        print(f"\n=== ANALYSE AUC-ROC ===")
        print(f"Runs avec AUC disponible: {len(auc_runs)}/{len(model_runs)}")

        # Impact des négations sur AUC
        if len(auc_runs[auc_runs['handle_negations'] == 'True']) > 0 and len(auc_runs[auc_runs['handle_negations'] == 'False']) > 0:
            auc_neg_true = auc_runs[auc_runs['handle_negations'] == 'True']['auc_score'].mean()
            auc_neg_false = auc_runs[auc_runs['handle_negations'] == 'False']['auc_score'].mean()
            print(f"\nImpact des négations sur AUC:")
            print(f"  - Avec négations: AUC moyen = {auc_neg_true:.4f}")
            print(f"  - Sans négations: AUC moyen = {auc_neg_false:.4f}")
            print(f"  - Différence: {auc_neg_true - auc_neg_false:+.4f}")

        # Impact des émotions sur AUC
        if len(auc_runs[auc_runs['handle_emotions'] == 'True']) > 0 and len(auc_runs[auc_runs['handle_emotions'] == 'False']) > 0:
            auc_emo_true = auc_runs[auc_runs['handle_emotions'] == 'True']['auc_score'].mean()
            auc_emo_false = auc_runs[auc_runs['handle_emotions'] == 'False']['auc_score'].mean()
            print(f"\nImpact des émotions sur AUC:")
            print(f"  - Avec émotions: AUC moyen = {auc_emo_true:.4f}")
            print(f"  - Sans émotions: AUC moyen = {auc_emo_false:.4f}")
            print(f"  - Différence: {auc_emo_true - auc_emo_false:+.4f}")

        # Performance AUC par technique
        auc_technique_performance = auc_runs.groupby('technique')['auc_score'].mean()
        print(f"\nPerformance AUC par technique:")
        for technique, score in auc_technique_performance.items():
            print(f"  - {technique.capitalize()}: AUC moyen = {score:.4f}")

        # Corrélation F1-Score vs AUC
        if len(auc_runs) > 2:
            correlation = auc_runs['f1_score'].corr(auc_runs['auc_score'])
            print(f"\nCorrélation F1-Score vs AUC-ROC: {correlation:.4f}")
            if correlation > 0.8:
                print("  → Forte corrélation positive: les deux métriques s'accordent")
            elif correlation > 0.5:
                print("  → Corrélation modérée: cohérence générale")
            else:
                print("  → Faible corrélation: les métriques divergent")

        # Meilleur modèle selon AUC
        best_auc_run = auc_runs.loc[auc_runs['auc_score'].idxmax()]
        print(f"\n🎯 Meilleur modèle selon AUC-ROC:")
        print(f"   - Technique: {best_auc_run['technique'].upper()}")
        print(f"   - Configuration: Neg={best_auc_run['handle_negations']}, Emo={best_auc_run['handle_emotions']}")
        print(f"   - AUC-ROC: {best_auc_run['auc_score']:.4f}")
        print(f"   - F1-Score: {best_auc_run['f1_score']:.4f}")

        # Comparer avec le meilleur F1
        best_f1_run = model_runs.loc[model_runs['f1_score'].idxmax()]
        if best_auc_run['run_id'] != best_f1_run['run_id']:
            print(f"\n⚠️  Le meilleur AUC diffère du meilleur F1:")
            print(f"   - Meilleur F1: {best_f1_run['technique'].upper()} (F1={best_f1_run['f1_score']:.4f})")
            print(f"   - Meilleur AUC: {best_auc_run['technique'].upper()} (AUC={best_auc_run['auc_score']:.4f})")
        else:
            print(f"\n✅ Le meilleur modèle est cohérent entre F1-Score et AUC-ROC")

    return df

if __name__ == "__main__":
    main()
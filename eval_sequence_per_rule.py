#!/usr/bin/env python3
"""Evaluate LSTM and Transformer models per-rule for F1 scores"""

import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report
from main import (
    create_robust_dataset, DistinctAlgebraicGNN, 
    train_model_with_strategy, evaluate_model,
    set_seed, ALGEBRAIC_RULES
)
from torch_geometric.data import DataLoader

def evaluate_sequence_per_rule(model_type, epochs=10):
    """Evaluate sequence model per-rule"""
    print(f"\n{'='*60}")
    print(f"🔬 {model_type.upper()} MODEL - PER-RULE EVALUATION")
    print(f"{'='*60}")
    
    # Create dataset
    dataset = create_robust_dataset()
    n_train = int(0.8 * len(dataset))
    train_data, val_data = dataset[:n_train], dataset[n_train:]
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Train model
    set_seed(42)
    model = DistinctAlgebraicGNN(encoder_type=model_type, use_uncertainty=False)
    model, _, _ = train_model_with_strategy(
        model, train_loader, val_loader, strategy="minimal", epochs=epochs, seed=42
    )
    
    # Evaluate
    metrics, y_true, y_pred, y_val_true, y_val_pred, eqs = evaluate_model(model, val_loader)
    
    # Per-rule F1 scores
    rule_names = list(ALGEBRAIC_RULES.values())
    per_rule_f1 = []
    
    for i in range(len(rule_names)):
        # Get predictions for this rule
        rule_mask = (y_true == i)
        if rule_mask.sum() > 0:
            rule_f1 = f1_score(y_true[rule_mask], y_pred[rule_mask], average='binary')
            per_rule_f1.append(rule_f1)
            print(f"{rule_names[i]}: F1 = {rule_f1:.4f}")
        else:
            per_rule_f1.append(0.0)
            print(f"{rule_names[i]}: F1 = 0.0000 (no samples)")
    
    print(f"\n📊 {model_type.upper()} Overall F1: {metrics['rule_f1']:.4f}")
    print(f"📊 {model_type.upper()} Overall Accuracy: {metrics['rule_accuracy']:.4f}")
    
    return per_rule_f1, metrics

if __name__ == "__main__":
    print("🔄 Evaluating sequence models per-rule...")
    
    # Evaluate LSTM
    lstm_f1s, lstm_metrics = evaluate_sequence_per_rule('lstm', epochs=10)
    
    # Evaluate Transformer  
    transformer_f1s, transformer_metrics = evaluate_sequence_per_rule('transformer', epochs=10)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY - PER-RULE F1 SCORES")
    print(f"{'='*80}")
    
    rule_names = list(ALGEBRAIC_RULES.values())
    print(f"{'Rule':<20} {'LSTM F1':<10} {'Transformer F1':<15}")
    print("-" * 50)
    
    for i, rule in enumerate(rule_names):
        print(f"{rule:<20} {lstm_f1s[i]:<10.4f} {transformer_f1s[i]:<15.4f}")
    
    print(f"\n📊 LSTM Overall: F1={lstm_metrics['rule_f1']:.4f}, Acc={lstm_metrics['rule_accuracy']:.4f}")
    print(f"📊 Transformer Overall: F1={transformer_metrics['rule_f1']:.4f}, Acc={transformer_metrics['rule_accuracy']:.4f}")
    
    # Save results for graph generation
    results = {
        'lstm_per_rule_f1': lstm_f1s,
        'transformer_per_rule_f1': transformer_f1s,
        'lstm_overall': lstm_metrics,
        'transformer_overall': transformer_metrics
    }
    
    import json
    with open('results/sequence_per_rule_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: results/sequence_per_rule_results.json")
    print("✅ Ready to update generate_per_rule_f1_graph.py with these values!")

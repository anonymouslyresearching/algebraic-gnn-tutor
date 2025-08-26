"""
ALGEBRAIC REASONING GNN TUTOR - COMPREHENSIVE RESEARCH PIPELINE
===============================================================

- Multi-seed, k-fold cross-validation (default: 3 seeds, 5 folds)
- Bootstrap confidence intervals for all key metrics
- Three distinct ablation architectures (GAT+Transformer+Uncertainty, GCN, MLP)
- Robust synthetic dataset (optionally: real-world data)
- Research-quality metrics, tables, and plots
- Per-class metrics, confusion matrices, ROC/PR curves, error analysis
- Ready for Google Colab and research applications

USAGE:
    - Run in Colab or locally (GPU recommended)
    - To add real-world data, see the 'load_real_world_data' function
"""

import warnings
warnings.filterwarnings('ignore')

import random
import sympy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool, GCNConv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import os

# =========================
# SETUP & REPRODUCIBILITY
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n🖥️  Device in use: {device}\n")

# =========================
# SYMBOLIC MATH UTILITIES
# =========================
x = sp.symbols('x')
NODE_TYPES = {'Symbol': 0, 'Integer': 1, 'Rational': 2, 'Add': 3, 'Mul': 4, 'Pow': 5, 'Eq': 6, 'Float': 7}
UNK_NODE = len(NODE_TYPES)

def create_ast_graph(expr):
    try:
        nodes, depths, edges = [], [], []
        def build_graph(e, depth):
            idx = len(nodes)
            node_type = NODE_TYPES.get(type(e).__name__, UNK_NODE)
            nodes.append(node_type)
            depths.append(min(depth, 19))
            for child in getattr(e, 'args', []):
                child_idx = build_graph(child, depth + 1)
                edges.extend([(idx, child_idx), (child_idx, idx)])
            return idx
        build_graph(expr, 0)
        if not edges: edges = [(0, 0)]
        return Data(
            x=torch.tensor(list(zip(nodes, depths)), dtype=torch.long),
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous()
        )
    except Exception as e:
        print(f"⚠️  AST graph creation failed: {e}")
        return Data(
            x=torch.tensor([[UNK_NODE, 0]], dtype=torch.long),
            edge_index=torch.tensor([[0], [0]], dtype=torch.long)
        )

ALGEBRAIC_RULES = {
    0: 'add_const', 1: 'sub_const', 2: 'div_coeff', 3: 'mul_denom',
    4: 'expand', 5: 'factor', 6: 'pow_reduce', 7: 'combine_fracs', 8: 'sub_var'
}

# =========================
# DATASET GENERATION
# =========================
def create_robust_dataset(num_per_rule=250, num_neg=600):
    print("🔄 Generating synthetic algebraic equation dataset...")
    dataset = []
    for rule_id in range(9):
        for _ in range(num_per_rule):
            try:
                if rule_id == 0:  # add_const: ax + b = c
                    a, b = random.randint(2, 6), random.randint(1, 8)
                    c = a * random.randint(1, 5) + b
                    eq = sp.Eq(a*x + b, c)
                    target_node = 1
                elif rule_id == 1:  # sub_const: ax - b = c
                    a, b = random.randint(2, 6), random.randint(1, 8)
                    c = a * random.randint(1, 5) - b
                    eq = sp.Eq(a*x - b, c)
                    target_node = 1
                elif rule_id == 2:  # div_coeff: ax = c
                    a = random.randint(2, 8)
                    c = a * random.randint(2, 6)
                    eq = sp.Eq(a*x, c)
                    target_node = 0
                elif rule_id == 3:  # mul_denom: x/d = c
                    d = random.randint(2, 5)
                    c = random.randint(2, 8)
                    eq = sp.Eq(x/d, c)
                    target_node = 0
                elif rule_id == 4:  # expand: (x+a)(x+b) = c
                    a, b = random.randint(1, 4), random.randint(1, 4)
                    c = random.randint(10, 30)
                    eq = sp.Eq((x+a)*(x+b), c)
                    target_node = 0
                elif rule_id == 5:  # factor: x^2 + px + q = 0
                    a, b = random.randint(1, 4), random.randint(1, 4)
                    eq = sp.Eq(x**2 + (a+b)*x + a*b, 0)
                    target_node = 0
                elif rule_id == 6:  # pow_reduce: x^n = c
                    n = random.randint(2, 4)
                    c = random.randint(8, 64)
                    eq = sp.Eq(x**n, c)
                    target_node = 0
                elif rule_id == 7:  # combine_fracs: a/b + c/d = e
                    a, b = random.randint(1, 4), random.randint(2, 5)
                    c, d = random.randint(1, 4), random.randint(2, 5)
                    e = random.randint(1, 5)
                    eq = sp.Eq(sp.Rational(a, b) + sp.Rational(c, d), e)
                    target_node = 0
                elif rule_id == 8:  # sub_var: ax + b = cx + d
                    a, c = random.sample(range(2, 7), 2)
                    b, d = random.randint(1, 8), random.randint(1, 8)
                    eq = sp.Eq(a*x + b, c*x + d)
                    target_node = 0
                else:
                    continue
                graph = create_ast_graph(eq)
                graph.y_rule = torch.tensor([rule_id], dtype=torch.long)
                graph.y_ptr = torch.tensor([min(target_node, len(graph.x)-1)], dtype=torch.long)
                graph.y_val = torch.tensor([1.0], dtype=torch.float)
                graph.equation_str = str(eq)
                dataset.append(graph)
            except Exception as e:
                print(f"⚠️  Example generation failed: {e}")
                continue
    # Add negative examples
    for _ in range(num_neg):
        try:
            graph = random.choice(dataset)
            new_graph = Data(x=graph.x.clone(), edge_index=graph.edge_index.clone())
            new_graph.y_rule = torch.tensor([random.randint(0, 8)], dtype=torch.long)
            new_graph.y_ptr = torch.tensor([random.randint(0, len(graph.x)-1)], dtype=torch.long)
            new_graph.y_val = torch.tensor([0.0], dtype=torch.float)
            new_graph.equation_str = "corrupted_" + getattr(graph, "equation_str", "")
            dataset.append(new_graph)
        except Exception as e:
            print(f"⚠️  Negative example generation failed: {e}")
            continue
    random.shuffle(dataset)
    print(f"✅ Dataset created with {len(dataset)} examples ({num_per_rule*9} positive, {num_neg} negative)")
    return dataset

# =========================
# (OPTIONAL) REAL-WORLD DATA
# =========================
def load_real_world_data():
    """
    Placeholder for loading real-world algebra equation data.
    Should return a list of Data objects with .y_rule, .y_ptr, .y_val, .equation_str
    """
    print("ℹ️  Real-world data loader not implemented. Using synthetic data.")
    return []

# =========================
# DISTINCT MODEL ARCHITECTURES
# =========================

class MainModelEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(UNK_NODE + 1, 32)
        self.depth_emb = nn.Embedding(20, 32)
        self.proj = nn.Linear(64, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim//4, heads=4, dropout=0.1) for _ in range(3)
        ])
        self.transformer = TransformerConv(hidden_dim, hidden_dim//4, heads=4, dropout=0.1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_emb = self.emb(x[:, 0])
        depth_emb = self.depth_emb(torch.clamp(x[:, 1], 0, 19))
        h = self.proj(torch.cat([node_emb, depth_emb], dim=1))
        for gat in self.gat_layers:
            h_new = F.elu(gat(h, edge_index))
            h = h + h_new
        h_trans = F.elu(self.transformer(h, edge_index))
        h = self.layer_norm(h + h_trans)
        data.node_embeddings = h
        return global_mean_pool(h, batch)

class SimpleGCNEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(UNK_NODE + 1, 16)
        self.depth_emb = nn.Embedding(20, 16)
        self.proj = nn.Linear(32, hidden_dim//2)
        self.gcn_layers = nn.ModuleList([
            GCNConv(hidden_dim//2, hidden_dim//2) for _ in range(2)
        ])
        self.final_proj = nn.Linear(hidden_dim//2, hidden_dim)
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_emb = self.emb(x[:, 0])
        depth_emb = self.depth_emb(torch.clamp(x[:, 1], 0, 19))
        h = self.proj(torch.cat([node_emb, depth_emb], dim=1))
        for gcn in self.gcn_layers:
            h = F.relu(gcn(h, edge_index))
            h = F.dropout(h, p=0.2, training=self.training)
        h = self.final_proj(h)
        data.node_embeddings = h
        return global_mean_pool(h, batch)

class MinimalMLPEncoder(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(UNK_NODE + 1, 8)
        self.depth_emb = nn.Embedding(20, 8)
        self.mlp = nn.Sequential(
            nn.Linear(16, hidden_dim//4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//4, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )
    def forward(self, data):
        x, batch = data.x, data.batch
        node_emb = self.emb(x[:, 0])
        depth_emb = self.depth_emb(torch.clamp(x[:, 1], 0, 19))
        h = torch.cat([node_emb, depth_emb], dim=1)
        h = self.mlp(h)
        data.node_embeddings = h
        return global_mean_pool(h, batch)

class LSTMSequenceEncoder(nn.Module):
    def __init__(self, hidden_dim=128, vocab_size=1000, embedding_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Better tokenization with character embeddings
        self.char_embedding = nn.Embedding(128, embedding_dim)  # ASCII characters
        
        # Improved LSTM with multiple layers and dropout
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim, 
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        
        # Project to 128 dimensions for main model compatibility
        self.projection = nn.Linear(hidden_dim * 2, 128)
        
        # Output heads with better architecture
        self.rule_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, len(ALGEBRAIC_RULES))
        )
        self.validity_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.pointer_network = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, data):
        # Handle batch vs single sample
        if hasattr(data, 'equation_str') and isinstance(data.equation_str, list):
            # Batch mode - multiple equations
            batch_size = len(data.equation_str)
            batch_embeddings = []
            
            for i in range(batch_size):
                eq_str = str(data.equation_str[i])
                
                # Better character tokenization
                tokens = []
                for c in eq_str[:30]:  # Increased length limit
                    tokens.append(min(ord(c), 127))  # Full ASCII range
                
                # Pad to fixed length
                while len(tokens) < 30:
                    tokens.append(0)
                
                tokens = torch.tensor(tokens, dtype=torch.long)
                
                # Character embeddings
                char_emb = self.char_embedding(tokens)  # [30, embedding_dim]
                
                # LSTM processing with attention
                lstm_out, (h_n, c_n) = self.lstm(char_emb.unsqueeze(0))  # [1, 30, hidden_dim*2]
                
                # Apply attention
                attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
                
                # Global average pooling with attention
                final_hidden = torch.mean(attended.squeeze(0), dim=0)  # [hidden_dim*2]
                
                # Project to 128 dimensions
                projected_128 = self.projection(final_hidden)  # [128]
                batch_embeddings.append(projected_128)
            
            # Stack batch embeddings
            batch_embeddings = torch.stack(batch_embeddings)  # [batch_size, 128]
            
            # Set node embeddings for the main model
            data.node_embeddings = batch_embeddings
            
            return batch_embeddings
        else:
            # Single sample mode
            eq_str = str(getattr(data, 'equation_str', 'x = 0'))
            
            # Better character tokenization
            tokens = []
            for c in eq_str[:30]:  # Increased length limit
                tokens.append(min(ord(c), 127))  # Full ASCII range
            
            # Pad to fixed length
            while len(tokens) < 30:
                tokens.append(0)
            
            tokens = torch.tensor(tokens, dtype=torch.long)
            
            # Character embeddings
            char_emb = self.char_embedding(tokens)  # [30, embedding_dim]
            
            # LSTM processing with attention
            lstm_out, (h_n, c_n) = self.lstm(char_emb.unsqueeze(0))  # [1, 30, hidden_dim*2]
            
            # Apply attention
            attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
            # Global average pooling with attention
            final_hidden = torch.mean(attended.squeeze(0), dim=0)  # [hidden_dim*2]
            
            # Project to 128 dimensions
            projected_128 = self.projection(final_hidden)  # [128]
            
            # Set node embeddings for the main model
            data.node_embeddings = projected_128.unsqueeze(0)  # [1, 128]
            
            return projected_128.unsqueeze(0)  # [1, 128]

class TransformerSequenceEncoder(nn.Module):
    def __init__(self, hidden_dim=128, vocab_size=1000, embedding_dim=64, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Better character embeddings
        self.char_embedding = nn.Embedding(128, embedding_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(50, embedding_dim))
        
        # Improved transformer with proper dimensions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=8, 
            dim_feedforward=hidden_dim * 2, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Project to 128 dimensions for main model compatibility
        self.projection = nn.Linear(embedding_dim, 128)
        
        # Better output heads
        self.rule_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, len(ALGEBRAIC_RULES))
        )
        self.validity_classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.pointer_network = nn.Linear(embedding_dim, 1)
    
    def forward(self, data):
        # Handle batch vs single sample
        if hasattr(data, 'equation_str') and isinstance(data.equation_str, list):
            # Batch mode - multiple equations
            batch_size = len(data.equation_str)
            batch_embeddings = []
            
            for i in range(batch_size):
                eq_str = str(data.equation_str[i])
                
                # Better character tokenization
                tokens = []
                for c in eq_str[:30]:  # Increased length limit
                    tokens.append(min(ord(c), 127))  # Full ASCII range
                
                # Pad to fixed length
                while len(tokens) < 30:
                    tokens.append(0)
                
                tokens = torch.tensor(tokens, dtype=torch.long)
                
                # Character embeddings with positional encoding
                char_emb = self.char_embedding(tokens)  # [30, embedding_dim]
                char_emb = char_emb + self.pos_encoding[:len(tokens)]  # Add positional encoding
                
                # Transformer processing
                transformed = self.transformer(char_emb.unsqueeze(0))  # [1, 30, embedding_dim]
                
                # Global average pooling
                pooled = torch.mean(transformed.squeeze(0), dim=0)  # [embedding_dim]
                
                # Project to 128 dimensions
                projected_128 = self.projection(pooled)  # [128]
                batch_embeddings.append(projected_128)
            
            # Stack batch embeddings
            batch_embeddings = torch.stack(batch_embeddings)  # [batch_size, 128]
            
            # Set node embeddings for the main model
            data.node_embeddings = batch_embeddings
            
            return batch_embeddings
        else:
            # Single sample mode
            eq_str = str(getattr(data, 'equation_str', 'x = 0'))
            
            # Better character tokenization
            tokens = []
            for c in eq_str[:30]:  # Increased length limit
                tokens.append(min(ord(c), 127))  # Full ASCII range
            
            # Pad to fixed length
            while len(tokens) < 30:
                tokens.append(0)
            
            tokens = torch.tensor(tokens, dtype=torch.long)
            
            # Character embeddings with positional encoding
            char_emb = self.char_embedding(tokens)  # [30, embedding_dim]
            char_emb = char_emb + self.pos_encoding[:len(tokens)]  # Add positional encoding
            
            # Transformer processing
            transformed = self.transformer(char_emb.unsqueeze(0))  # [1, 30, embedding_dim]
            
            # Global average pooling
            pooled = torch.mean(transformed.squeeze(0), dim=0)  # [embedding_dim]
            
            # Project to 128 dimensions
            projected_128 = self.projection(pooled)  # [128]
            
            # Set node embeddings for the main model
            data.node_embeddings = projected_128.unsqueeze(0)  # [1, 128]
            
            return projected_128.unsqueeze(0)  # [1, 128]
    
    def _tokenize_equation(self, equation_str):
        # Simple character-level tokenization
        tokens = []
        for c in str(equation_str)[:20]:  # Limit to 20 characters
            tokens.append(ord(c) % 64)  # Use smaller range
        
        # Pad to fixed length
        while len(tokens) < 20:
            tokens.append(0)
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens.unsqueeze(0)  # [1, 20]

class DistinctAlgebraicGNN(nn.Module):
    def __init__(self, encoder_type="main", use_uncertainty=True):
        super().__init__()
        self.use_uncertainty = use_uncertainty
        if encoder_type == "main":
            self.encoder = MainModelEncoder()
        elif encoder_type == "simple":
            self.encoder = SimpleGCNEncoder()
        elif encoder_type == "minimal":
            self.encoder = MinimalMLPEncoder()
        elif encoder_type == "lstm":
            self.encoder = LSTMSequenceEncoder()
        elif encoder_type == "transformer":
            self.encoder = TransformerSequenceEncoder()
        else:
            self.encoder = MinimalMLPEncoder()
        head_size = 64 if encoder_type in ["minimal", "lstm", "transformer"] else 128
        self.rule_head = nn.Sequential(
            nn.Linear(128, head_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_size, 9)
        )
        self.val_head = nn.Sequential(
            nn.Linear(128, head_size//2),
            nn.ReLU(),
            nn.Linear(head_size//2, 1)
        )
        self.ptr_head = nn.Linear(128, 128)
        if use_uncertainty:
            self.log_var_rule = nn.Parameter(torch.zeros(()))
            self.log_var_val = nn.Parameter(torch.zeros(()))
            self.log_var_ptr = nn.Parameter(torch.zeros(()))
    def forward(self, data):
        graph_emb = self.encoder(data)
        rule_logits = self.rule_head(graph_emb)
        val_logits = self.val_head(graph_emb)
        val_probs = torch.sigmoid(val_logits).squeeze(-1)
        query = self.ptr_head(graph_emb)
        
        # Handle sequence models (single node) vs graph models (multiple nodes)
        if hasattr(data, 'batch') and data.batch is not None and len(data.batch) > 1:
            # Graph model with multiple nodes
            query_exp = query[data.batch]
            ptr_scores = torch.sum(query_exp * data.node_embeddings, dim=-1)
        else:
            # Sequence model with single node
            ptr_scores = torch.sum(query * data.node_embeddings, dim=-1)
        
        return rule_logits, val_probs, ptr_scores

# =========================
# TRAINING & EVALUATION
# =========================

def train_model_with_strategy(model, train_loader, val_loader, strategy="main", epochs=15, seed=42):
    set_seed(seed)
    model = model.to(device)
    if strategy == "main":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_weights = (1.0, 1.0, 1.0)
    elif strategy == "simple":
        optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
        loss_weights = (2.0, 0.5, 1.0)
    elif strategy == "minimal":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=2e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        loss_weights = (1.0, 2.0, 0.5)
    else:
        # Default strategy for sequence models
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
        loss_weights = (1.0, 1.0, 1.0)
    best_f1 = 0
    patience = 5
    no_improvement = 0
    train_history = []
    val_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            rule_logits, val_probs, ptr_scores = model(batch)
            rule_loss = F.cross_entropy(rule_logits, batch.y_rule.squeeze())
            val_loss = F.binary_cross_entropy(val_probs, batch.y_val)
            ptr_loss = 0
            batch_size = rule_logits.size(0)
            for i in range(batch_size):
                mask = batch.batch == i
                if mask.sum() > 0:
                    scores = ptr_scores[mask]
                    target = batch.y_ptr[i].item()
                    if 0 <= target < len(scores):
                        ptr_loss += F.cross_entropy(scores.unsqueeze(0), torch.tensor([target], device=device))
            ptr_loss /= batch_size
            if model.use_uncertainty and hasattr(model, 'log_var_rule'):
                total_loss = (
                    torch.exp(-model.log_var_rule) * rule_loss + model.log_var_rule +
                    torch.exp(-model.log_var_val) * val_loss + model.log_var_val +
                    torch.exp(-model.log_var_ptr) * ptr_loss + model.log_var_ptr
                )
            else:
                total_loss = (loss_weights[0] * rule_loss +
                              loss_weights[1] * val_loss +
                              loss_weights[2] * ptr_loss)
            optimizer.zero_grad()
            total_loss.backward()
            if strategy == "main":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            elif strategy == "simple":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += total_loss.item()
        scheduler.step()
        model.eval()
        val_metrics, _, _, _, _, _ = evaluate_model(model, val_loader)  # Fixed: unpack 6 values
        train_history.append(epoch_loss / len(train_loader))
        val_history.append(val_metrics)
        print(f"    Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f} | Val F1: {val_metrics['rule_f1']:.4f}")
        if val_metrics['rule_f1'] > best_f1:
            best_f1 = val_metrics['rule_f1']
            no_improvement = 0
            torch.save(model.state_dict(), f'{strategy}_model_best.pth')
        else:
            no_improvement += 1
        if no_improvement >= patience:
            print(f"    Early stopping at epoch {epoch}")
            break
    model.load_state_dict(torch.load(f'{strategy}_model_best.pth'))
    return model, train_history, val_history

def evaluate_model(model, data_loader):
    model.eval()
    all_rule_true, all_rule_pred = [], []
    all_val_true, all_val_pred = [], []
    all_ptr_ranks = []
    all_eqs = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            rule_logits, val_probs, ptr_scores = model(batch)
            rule_pred = torch.argmax(rule_logits, dim=1)
            all_rule_true.extend(batch.y_rule.squeeze().cpu().tolist())
            all_rule_pred.extend(rule_pred.cpu().tolist())
            all_val_true.extend(batch.y_val.cpu().tolist())
            all_val_pred.extend(val_probs.cpu().tolist())
            batch_size = rule_logits.size(0)
            for i in range(batch_size):
                mask = batch.batch == i
                if mask.sum() > 0:
                    scores = ptr_scores[mask].cpu().tolist()
                    target = batch.y_ptr[i].item()
                    if 0 <= target < len(scores):
                        sorted_idx = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
                        rank = sorted_idx.index(target) + 1
                        all_ptr_ranks.append(rank)
            if hasattr(batch, "equation_str"):
                all_eqs.extend(batch.equation_str)
            else:
                all_eqs.extend([""]*batch_size)
    metrics = {
        'rule_accuracy': float(accuracy_score(all_rule_true, all_rule_pred)),
        'rule_f1': float(precision_recall_fscore_support(all_rule_true, all_rule_pred, average='macro', zero_division=0)[2]),
        'validity_auc': float(roc_auc_score(all_val_true, all_val_pred)) if len(set(all_val_true)) > 1 else 0.5,
        'pointer_mrr': float(np.mean([1.0/r for r in all_ptr_ranks])) if all_ptr_ranks else 0.0,
        'pointer_top3': float(np.mean([r <= 3 for r in all_ptr_ranks])) if all_ptr_ranks else 0.0
    }
    return metrics, all_rule_true, all_rule_pred, all_val_true, all_val_pred, all_eqs

# =========================
# BOOTSTRAP CONFIDENCE INTERVALS
# =========================

def bootstrap_ci(metric_list, n_bootstrap=1000, ci=0.95):
    arr = np.array(metric_list)
    boot_means = []
    n = len(arr)
    for _ in range(n_bootstrap):
        sample = np.random.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, (1-ci)/2*100)
    upper = np.percentile(boot_means, (1+(ci))/2*100)
    return np.mean(arr), lower, upper

# =========================
# K-FOLD CROSS-VALIDATION + MULTI-SEED + CI
# =========================

def kfold_multiseed_eval(dataset, k=5, seeds=[42, 43, 44], epochs=10, batch_size=32):
    results = {}
    all_perclass = {}
    for model_type, strategy, use_uncertainty in [
        ("main", "main", True),
        ("simple", "simple", False),
        ("minimal", "minimal", False),
        ("lstm", "minimal", False),
        ("transformer", "minimal", False)
    ]:
        print(f"\n{'='*60}\n🧪 {model_type.upper()} MODEL: MULTI-SEED, {k}-FOLD CROSS-VALIDATION\n{'='*60}")
        all_metrics = {'rule_f1': [], 'rule_accuracy': [], 'validity_auc': [], 'pointer_mrr': [], 'pointer_top3': []}
        all_perclass_metrics = []
        for seed in seeds:
            set_seed(seed)
            kf = KFold(n_splits=k, shuffle=True, random_state=seed)
            for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                print(f"  Seed {seed} | Fold {fold+1}/{k}")
                train_data = [dataset[i] for i in train_idx]
                val_data = [dataset[i] for i in val_idx]
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
                model = DistinctAlgebraicGNN(encoder_type=model_type, use_uncertainty=use_uncertainty)
                model, _, _ = train_model_with_strategy(
                    model, train_loader, val_loader, strategy=strategy, epochs=epochs, seed=seed+fold
                )
                metrics, y_true, y_pred, y_val_true, y_val_pred, _ = evaluate_model(model, val_loader)
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
                # Per-class F1
                perclass = precision_recall_fscore_support(y_true, y_pred, labels=list(range(9)), zero_division=0)
                all_perclass_metrics.append(perclass[2])  # F1 per class
        # Bootstrap CIs
        summary = {}
        for key in all_metrics:
            mean, lower, upper = bootstrap_ci(all_metrics[key])
            summary[key+"_mean"] = mean
            summary[key+"_ci_lower"] = lower
            summary[key+"_ci_upper"] = upper
        # Per-class F1 mean/std
        all_perclass_metrics = np.array(all_perclass_metrics)
        perclass_mean = np.mean(all_perclass_metrics, axis=0)
        perclass_std = np.std(all_perclass_metrics, axis=0)
        summary['perclass_f1_mean'] = perclass_mean.tolist()
        summary['perclass_f1_std'] = perclass_std.tolist()
        print(f"\n📊 {model_type} model: Macro F1 mean={summary['rule_f1_mean']:.4f}, 95% CI=({summary['rule_f1_ci_lower']:.4f}, {summary['rule_f1_ci_upper']:.4f})")
        results[model_type] = summary
        all_perclass[model_type] = (perclass_mean, perclass_std)
    return results, all_perclass

# =========================
# ADVANCED EVALUATION & PLOTS
# =========================

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(9)))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(ALGEBRAIC_RULES.values()),
                yticklabels=list(ALGEBRAIC_RULES.values()))
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'figures/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_auc(y_true, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.title(f'ROC Curve: {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'figures/roc_curve_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pr_curve(y_true, y_score, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.figure()
    plt.plot(recall, precision, label='PR curve')
    plt.title(f'Precision-Recall Curve: {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'figures/pr_curve_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

def show_errors(y_true, y_pred, eqs, n=5):
    errors = [(i, t, p) for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    print(f"\n🔍 Showing {n} random errors:")
    for idx, t, p in random.sample(errors, min(n, len(errors))):
        print(f"  Example {idx}: True={ALGEBRAIC_RULES[t]}, Pred={ALGEBRAIC_RULES[p]}, Eq={eqs[idx]}")

# =========================
# MAIN EXPERIMENT PIPELINE
# =========================

def run_experiment(
    use_real_world=False, k=5, seeds=[42,43,44], epochs=10, batch_size=32
):
    print("="*80)
    print("ALGEBRAIC REASONING GNN - COMPREHENSIVE EVALUATION")
    print("MULTI-SEED, K-FOLD CROSS-VALIDATION, ADVANCED EVALUATION")
    print("="*80)
    
    # Create figures directory
    os.makedirs('figures', exist_ok=True)
    
    # Data
    if use_real_world:
        dataset = load_real_world_data()
        if not dataset:
            print("No real-world data found, using synthetic.")
            dataset = create_robust_dataset()
    else:
        dataset = create_robust_dataset()
    # K-Fold Cross-Validation
    results, all_perclass = kfold_multiseed_eval(dataset, k=k, seeds=seeds, epochs=epochs, batch_size=batch_size)
    # Final single split for advanced evaluation/plots
    n_train = int(0.8 * len(dataset))
    train_data, val_data = dataset[:n_train], dataset[n_train:]
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    # Train and evaluate each model for advanced metrics/plots
    all_eval = {}
    for model_type, strategy, use_uncertainty in [
        ("main", "main", True),
        ("simple", "simple", False),
        ("minimal", "minimal", False),
        ("lstm", "minimal", False),
        ("transformer", "minimal", False)
    ]:
        print(f"\n{'='*60}\n🔬 {model_type.upper()} MODEL - ADVANCED EVALUATION\n{'='*60}")
        set_seed(42)
        model = DistinctAlgebraicGNN(encoder_type=model_type, use_uncertainty=use_uncertainty)
        model, _, _ = train_model_with_strategy(
            model, train_loader, val_loader, strategy=strategy, epochs=epochs, seed=42
        )
        metrics, y_true, y_pred, y_val_true, y_val_pred, eqs = evaluate_model(model, val_loader)
        all_eval[model_type] = (metrics, y_true, y_pred, y_val_true, y_val_pred, eqs)
        print(f"\n📋 {model_type.upper()} MODEL - Per-class Metrics:")
        print(classification_report(y_true, y_pred, target_names=list(ALGEBRAIC_RULES.values()), digits=4))
        plot_confusion_matrix(y_true, y_pred, model_type)
        plot_roc_auc(y_val_true, y_val_pred, model_type)
        plot_pr_curve(y_val_true, y_val_pred, model_type)
        show_errors(y_true, y_pred, eqs, n=5)
    # Summary Table
    summary = []
    for model_type in ["main", "simple", "minimal", "lstm", "transformer"]:
        row = {
            "Model": model_type,
            "F1 mean": results[model_type]["rule_f1_mean"],
            "F1 95% CI lower": results[model_type]["rule_f1_ci_lower"],
            "F1 95% CI upper": results[model_type]["rule_f1_ci_upper"],
            "Accuracy mean": results[model_type]["rule_accuracy_mean"],
            "Accuracy 95% CI lower": results[model_type]["rule_accuracy_ci_lower"],
            "Accuracy 95% CI upper": results[model_type]["rule_accuracy_ci_upper"],
            "AUC mean": results[model_type]["validity_auc_mean"],
            "AUC 95% CI lower": results[model_type]["validity_auc_ci_lower"],
            "AUC 95% CI upper": results[model_type]["validity_auc_ci_upper"],
            "Pointer MRR mean": results[model_type]["pointer_mrr_mean"],
            "Pointer MRR 95% CI lower": results[model_type]["pointer_mrr_ci_lower"],
            "Pointer MRR 95% CI upper": results[model_type]["pointer_mrr_ci_upper"],
            "Pointer Top3 mean": results[model_type]["pointer_top3_mean"],
            "Pointer Top3 95% CI lower": results[model_type]["pointer_top3_ci_lower"],
            "Pointer Top3 95% CI upper": results[model_type]["pointer_top3_ci_upper"],
        }
        summary.append(row)
    df = pd.DataFrame(summary)
    print("\n" + "="*80)
    print("SUMMARY TABLE (MEAN & 95% CI OVER SEEDS & FOLDS)")
    print("="*80)
    print(df.to_string(index=False, float_format='%.4f'))
    df.to_csv("results/model_summary.csv", index=False)
    # Save all results
    os.makedirs('results', exist_ok=True)
    with open("results/experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    # Save per-class F1s
    perclass_df = pd.DataFrame({
        "Rule": list(ALGEBRAIC_RULES.values()),
        "Main F1 mean": all_perclass["main"][0],
        "Main F1 std": all_perclass["main"][1],
        "Simple F1 mean": all_perclass["simple"][0],
        "Simple F1 std": all_perclass["simple"][1],
        "Minimal F1 mean": all_perclass["minimal"][0],
        "Minimal F1 std": all_perclass["minimal"][1],
        "LSTM F1 mean": all_perclass["lstm"][0] if "lstm" in all_perclass else [0]*9,
        "LSTM F1 std": all_perclass["lstm"][1] if "lstm" in all_perclass else [0]*9,
        "Transformer F1 mean": all_perclass["transformer"][0] if "transformer" in all_perclass else [0]*9,
        "Transformer F1 std": all_perclass["transformer"][1] if "transformer" in all_perclass else [0]*9,
    })
    perclass_df.to_csv("results/perclass_f1.csv", index=False)
    print("\n🎉 Experiment complete! All results saved to 'results/' directory.")
    print("✅ Results saved: model_summary.csv, experiment_results.json, perclass_f1.csv")
    print("Ready for further analysis and research applications.")
    return results

# =========================
# EXECUTION ENTRY POINT
# =========================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔬 ALGEBRAIC REASONING GNN")
    print("🚀 MULTI-SEED, K-FOLD, ADVANCED EVALUATION, CONFIDENCE INTERVALS")
    print("="*80)
    run_experiment()

# =========================
# COLAB CONVENIENCE
# =========================

def quick_run():
    return run_experiment()

print("\n" + "="*80)
print("🔬 ALGEBRAIC REASONING GNN")
print("📚 Multi-seed, k-fold, advanced evaluation, confidence intervals, real-world ready")
print("🚀 Run run_experiment() or quick_run() to start")
print("="*80)

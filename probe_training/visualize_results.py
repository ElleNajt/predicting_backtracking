"""Visualize probe training results."""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def create_heatmap(results_df, metric='val_f1'):
    """
    Create heatmap of probe performance across layers and lags.

    Args:
        results_df: DataFrame with columns ['layer', 'lag', metric]
        metric: Which metric to visualize

    Returns:
        plotly figure
    """
    # Pivot data for heatmap
    pivot_data = results_df.pivot(index='lag', columns='layer', values=metric)

    # Sort lags in descending order (most negative at top)
    pivot_data = pivot_data.sort_index(ascending=False)

    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='Viridis',
        colorbar=dict(title=metric.upper()),
        text=pivot_data.values,
        texttemplate='%{text:.3f}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title=f'Probe Performance: {metric.upper()} across Layers and Lags',
        xaxis_title='Layer',
        yaxis_title='Lag (negative = predict future)',
        height=500,
        width=700,
        xaxis=dict(tickmode='linear', tick0=8, dtick=2),
        yaxis=dict(tickmode='linear')
    )

    return fig


def create_line_plot(results_df, metric='val_f1'):
    """
    Create line plot showing how performance changes with lag for each layer.

    Args:
        results_df: DataFrame with columns ['layer', 'lag', metric]
        metric: Which metric to visualize

    Returns:
        plotly figure
    """
    fig = go.Figure()

    layers = sorted(results_df['layer'].unique())

    for layer in layers:
        layer_data = results_df[results_df['layer'] == layer].sort_values('lag')

        fig.add_trace(go.Scatter(
            x=layer_data['lag'],
            y=layer_data[metric],
            mode='lines+markers',
            name=f'Layer {layer}',
            line=dict(width=2),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title=f'{metric.upper()} vs Lag across Different Layers',
        xaxis_title='Lag (negative = predict future)',
        yaxis_title=metric.upper(),
        height=500,
        width=800,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Add vertical line at lag=0
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    return fig


def create_comparison_plot(results_df):
    """
    Create subplot comparing multiple metrics.

    Returns:
        plotly figure
    """
    metrics = ['val_f1', 'val_acc', 'val_auc']
    titles = ['F1 Score', 'Accuracy', 'AUROC']

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=titles,
        horizontal_spacing=0.1
    )

    layers = sorted(results_df['layer'].unique())
    colors = px.colors.qualitative.Plotly

    for i, metric in enumerate(metrics, 1):
        for j, layer in enumerate(layers):
            layer_data = results_df[results_df['layer'] == layer].sort_values('lag')

            showlegend = (i == 1)  # Only show legend for first subplot

            fig.add_trace(
                go.Scatter(
                    x=layer_data['lag'],
                    y=layer_data[metric],
                    mode='lines+markers',
                    name=f'Layer {layer}',
                    line=dict(color=colors[j % len(colors)], width=2),
                    marker=dict(size=6),
                    showlegend=showlegend
                ),
                row=1, col=i
            )

        # Add vertical line at lag=0
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3, row=1, col=i)

        fig.update_xaxes(title_text="Lag", row=1, col=i)
        fig.update_yaxes(title_text=titles[i-1], row=1, col=i)

    fig.update_layout(
        title_text="Probe Performance Comparison Across Metrics",
        height=400,
        width=1400,
        hovermode='x unified'
    )

    return fig


def create_class_balance_plot(results_df):
    """
    Visualize class balance (positive ratio) in training and validation sets.

    Returns:
        plotly figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Training Set', 'Validation Set']
    )

    layers = sorted(results_df['layer'].unique())
    colors = px.colors.qualitative.Plotly

    for j, layer in enumerate(layers):
        layer_data = results_df[results_df['layer'] == layer].sort_values('lag')

        # Training positive ratio
        fig.add_trace(
            go.Scatter(
                x=layer_data['lag'],
                y=layer_data['pos_ratio_train'],
                mode='lines+markers',
                name=f'Layer {layer}',
                line=dict(color=colors[j % len(colors)]),
                showlegend=True
            ),
            row=1, col=1
        )

        # Validation positive ratio
        fig.add_trace(
            go.Scatter(
                x=layer_data['lag'],
                y=layer_data['pos_ratio_val'],
                mode='lines+markers',
                name=f'Layer {layer}',
                line=dict(color=colors[j % len(colors)]),
                showlegend=False
            ),
            row=1, col=2
        )

    fig.update_xaxes(title_text="Lag", row=1, col=1)
    fig.update_xaxes(title_text="Lag", row=1, col=2)
    fig.update_yaxes(title_text="Positive Class Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Positive Class Ratio", row=1, col=2)

    fig.update_layout(
        title_text="Class Balance Across Configurations",
        height=400,
        width=1000,
        hovermode='x unified'
    )

    return fig


def main():
    """Generate all visualizations."""
    RESULTS_DIR = "/workspace/probe_training/results"
    results_path = os.path.join(RESULTS_DIR, "probe_results.csv")

    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    # Load results
    print(f"Loading results from {results_path}...")
    results_df = pd.read_csv(results_path)

    print(f"Loaded {len(results_df)} probe results")
    print(f"\nColumns: {results_df.columns.tolist()}")
    print(f"\nSummary statistics:")
    print(results_df[['val_f1', 'val_acc', 'val_auc']].describe())

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Heatmap for F1 score
    fig_heatmap_f1 = create_heatmap(results_df, metric='val_f1')
    fig_heatmap_f1.write_html(os.path.join(RESULTS_DIR, "heatmap_f1.html"))
    print("Saved: heatmap_f1.html")

    # 2. Heatmap for accuracy
    fig_heatmap_acc = create_heatmap(results_df, metric='val_acc')
    fig_heatmap_acc.write_html(os.path.join(RESULTS_DIR, "heatmap_accuracy.html"))
    print("Saved: heatmap_accuracy.html")

    # 3. Line plot for F1
    fig_line_f1 = create_line_plot(results_df, metric='val_f1')
    fig_line_f1.write_html(os.path.join(RESULTS_DIR, "lineplot_f1.html"))
    print("Saved: lineplot_f1.html")

    # 4. Comparison plot
    fig_comparison = create_comparison_plot(results_df)
    fig_comparison.write_html(os.path.join(RESULTS_DIR, "comparison_metrics.html"))
    print("Saved: comparison_metrics.html")

    # 5. Class balance plot
    fig_balance = create_class_balance_plot(results_df)
    fig_balance.write_html(os.path.join(RESULTS_DIR, "class_balance.html"))
    print("Saved: class_balance.html")

    print("\nAll visualizations saved!")

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    print("\nBest probe by F1 score:")
    best_row = results_df.loc[results_df['val_f1'].idxmax()]
    print(f"Layer {best_row['layer']}, Lag {best_row['lag']:+d}: F1={best_row['val_f1']:.4f}, Acc={best_row['val_acc']:.4f}")

    print("\nPerformance at lag=0 (current position):")
    lag0 = results_df[results_df['lag'] == 0].sort_values('val_f1', ascending=False)
    print(lag0[['layer', 'val_f1', 'val_acc', 'val_auc']].to_string(index=False))

    print("\nBest predictive performance (negative lags):")
    negative_lags = results_df[results_df['lag'] < 0].sort_values('val_f1', ascending=False).head(5)
    print(negative_lags[['layer', 'lag', 'val_f1', 'val_acc', 'val_auc']].to_string(index=False))


if __name__ == "__main__":
    main()

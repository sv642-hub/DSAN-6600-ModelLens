from modellens.analysis.logit_lens import run_logit_lens, decode_logit_lens
from modellens.analysis.attention import (
    run_attention_analysis,
    head_summary,
    compute_attention_pattern_metrics,
    run_comparative_attention,
)
from modellens.analysis.activation_patching import run_activation_patching
from modellens.analysis.embeddings import run_embeddings_analysis, nearest_neighbors
from modellens.analysis.residual_stream import (
    run_residual_analysis,
    identify_critical_layers,
)
from modellens.analysis.circuit_discovery import discover_circuit, summarize_circuit
from modellens.analysis.batch_patching import (
    run_batch_patching,
    summarize_batch_patching,
)
from modellens.analysis.forward_trace import (
    run_forward_trace,
    trace_token_position_norms,
)
from modellens.analysis.backward_trace import run_backward_trace
from modellens.analysis.comparison import (
    compare_forward_outputs,
    run_comparative_logit_lens,
    comparative_logit_lens_metrics,
)
from modellens.analysis.divergence import (
    run_activation_divergence,
    first_divergence_module,
)
from modellens.analysis.training_snapshots import TrainingSnapshot, SnapshotStore

__all__ = [
    # Logit lens
    "run_logit_lens",
    "decode_logit_lens",
    # Attention
    "run_attention_analysis",
    "head_summary",
    "compute_attention_pattern_metrics",
    "run_comparative_attention",
    # Activation patching
    "run_activation_patching",
    # Embeddings
    "run_embeddings_analysis",
    "nearest_neighbors",
    # Residual stream
    "run_residual_analysis",
    "identify_critical_layers",
    # Circuit discovery
    "discover_circuit",
    "summarize_circuit",
    # Forward trace
    "run_forward_trace",
    "trace_token_position_norms",
    # Backward trace
    "run_backward_trace",
    # Comparison
    "compare_forward_outputs",
    "run_comparative_logit_lens",
    "comparative_logit_lens_metrics",
    # Divergence
    "run_activation_divergence",
    "first_divergence_module",
    # Training snapshots
    "TrainingSnapshot",
    "SnapshotStore",
    # Batch patching
    "run_batch_patching",
    "summarize_batch_patching",
]

using Printf  # for @sprintf in LoggingCallback

# callbacks.jl — Built-in observer callbacks.
#
# Two concrete `AbstractCallback` subtypes ship in v0.2:
#
#   HistoryCallback   — accumulate per-iteration scalars into a vector
#   LoggingCallback   — print a table to an io stream
#
# Both observe-only — their `on_event!` returns `nothing`. They're independent
# of the stopping-criteria family (which also subtypes AbstractCallback but
# returns `(Bool, Symbol)` from `on_event!`).

# ============================================================================
# HistoryCallback
# ============================================================================

"""
    HISTORY_FIELDS

Available scalar fields for [`HistoryCallback`](@ref) and the default
column set for [`LoggingCallback`](@ref).

| Field      | Meaning                                          |
|------------|--------------------------------------------------|
| `:k`       | Outer iteration index                            |
| `:F_norm`  | `‖F(z_k)‖` (residual at last trial point)        |
| `:d_norm`  | `‖d_k‖`, current search direction               |
| `:α`       | Last accepted line-search step `α_k`            |
| `:n_evals` | Cumulative F evaluations                         |
| `:resid`   | `cache.resid` (`NaN` until set at termination)   |
| `:elapsed` | Wall-clock seconds since solve start             |

```jldoctest
julia> HISTORY_FIELDS
(:k, :F_norm, :d_norm, :α, :n_evals, :resid, :elapsed)
```
"""
const HISTORY_FIELDS = (:k, :F_norm, :d_norm, :α, :n_evals, :resid, :elapsed)

"""
    HistoryCallback(; fields = HISTORY_FIELDS)
    HistoryCallback(extractor::Function)

Per-iteration history accumulator. Two constructor paths:

- **Pre-declared field names** (the common case): pick any subset of
  [`HISTORY_FIELDS`](@ref). Each iter, `on_event!(cb, cache, :post_iter)`
  appends a NamedTuple with those fields to `cb.history`.
- **Custom extractor**: `HistoryCallback(extractor = cache -> NamedTuple)`
  bypasses `fields` and uses the user's function to produce each row.
  Useful for derived quantities (e.g. `max(abs, F)`, condition numbers).

Constructor errors if both `fields` and `extractor` are supplied.

# Example
```julia
hist = HistoryCallback(fields = (:k, :F_norm))
sol = solve(prob, DFProjection(callbacks = [hist]))
hist.history  # Vector{NamedTuple} with .k and .F_norm per row
```
"""
mutable struct HistoryCallback{E} <: AbstractCallback
    fields::Union{Nothing, Tuple}
    extractor::E
    history::Vector{Any}
end

function HistoryCallback(; fields = HISTORY_FIELDS, extractor = nothing)
    if extractor !== nothing && fields !== HISTORY_FIELDS
        # User supplied both — reject.
        error("HistoryCallback: pass either `fields` or `extractor`, not both.")
    end
    if extractor !== nothing
        return HistoryCallback{typeof(extractor)}(nothing, extractor, Any[])
    end
    # Validate fields against HISTORY_FIELDS
    fields_tuple = Tuple(fields)
    for f in fields_tuple
        if !(f in HISTORY_FIELDS)
            error("HistoryCallback: unknown field `:$f`. Available: $HISTORY_FIELDS")
        end
    end
    return HistoryCallback{Nothing}(fields_tuple, nothing, Any[])
end

# Allow positional extractor: HistoryCallback(cache -> ...)
HistoryCallback(extractor::Function) =
    HistoryCallback(; extractor = extractor)

@inline _extract_field(cache, ::Val{:k})       = cache.k
@inline _extract_field(cache, ::Val{:F_norm})  = _norm2(cache.Fz)
@inline _extract_field(cache, ::Val{:d_norm})  = _norm2(cache.d)
@inline _extract_field(cache, ::Val{:α})       = cache.α_prev
@inline _extract_field(cache, ::Val{:n_evals}) = cache.n_evals
@inline _extract_field(cache, ::Val{:resid})   = cache.resid
@inline _extract_field(cache, ::Val{:elapsed}) = time() - cache.t_start

function on_event!(cb::HistoryCallback, cache, event::Symbol)
    event === :post_iter || return nothing
    if cb.extractor !== nothing
        push!(cb.history, cb.extractor(cache))
    else
        row = NamedTuple{cb.fields}(map(f -> _extract_field(cache, Val(f)), cb.fields))
        push!(cb.history, row)
    end
    return nothing
end

# ============================================================================
# LoggingCallback
# ============================================================================

"""
    LoggingCallback(; io = stdout, columns = HISTORY_FIELDS, every = 1,
                      header_every = 50, footer = true)

Per-iteration table logger. Prints:

- A header row at `:initialize`
- One row per iteration at `:post_iter`, respecting `every` (1 = every iter,
  10 = every 10th iter)
- Header re-prints every `header_every` rows (set to `0` to disable)
- A footer summary at `:terminate` (skip via `footer = false`)
"""
mutable struct LoggingCallback{IO_T} <: AbstractCallback
    io::IO_T
    columns::Tuple
    every::Int
    header_every::Int
    footer::Bool
    row_count::Int
end

function LoggingCallback(; io::IO = stdout,
                          columns = HISTORY_FIELDS,
                          every::Int = 1,
                          header_every::Int = 50,
                          footer::Bool = true)
    cols = Tuple(columns)
    for f in cols
        if !(f in HISTORY_FIELDS)
            error("LoggingCallback: unknown column `:$f`. Available: $HISTORY_FIELDS")
        end
    end
    return LoggingCallback{typeof(io)}(io, cols, every, header_every, footer, 0)
end

@inline _col_width(::Val{:k})       = 6
@inline _col_width(::Val{:F_norm})  = 12
@inline _col_width(::Val{:d_norm})  = 12
@inline _col_width(::Val{:α})       = 10
@inline _col_width(::Val{:n_evals}) = 8
@inline _col_width(::Val{:resid})   = 12
@inline _col_width(::Val{:elapsed}) = 10

@inline _col_format(cache, ::Val{:k})       = lpad(string(cache.k), 6)
@inline _col_format(cache, ::Val{:F_norm})  = lpad(_fmt_sci(_norm2(cache.Fz)), 12)
@inline _col_format(cache, ::Val{:d_norm})  = lpad(_fmt_sci(_norm2(cache.d)), 12)
@inline _col_format(cache, ::Val{:α})       = lpad(_fmt_dec(cache.α_prev), 10)
@inline _col_format(cache, ::Val{:n_evals}) = lpad(string(cache.n_evals), 8)
@inline _col_format(cache, ::Val{:resid})   = lpad(_fmt_sci(cache.resid), 12)
@inline _col_format(cache, ::Val{:elapsed}) = lpad(_fmt_dec(time() - cache.t_start) * "s", 10)

_fmt_sci(x::Real) = isnan(x) ? "NaN" : isinf(x) ? "Inf" : (@sprintf "%.2e" x)
_fmt_dec(x::Real) = isnan(x) ? "NaN" : isinf(x) ? "Inf" : (@sprintf "%.3f" x)

function _print_header(cb::LoggingCallback)
    parts = map(col -> lpad(string(col), _col_width(Val(col))), cb.columns)
    println(cb.io, join(parts, "  "))
    return nothing
end

function _print_row(cb::LoggingCallback, cache)
    parts = map(col -> _col_format(cache, Val(col)), cb.columns)
    println(cb.io, join(parts, "  "))
    return nothing
end

function on_event!(cb::LoggingCallback, cache, event::Symbol)
    if event === :initialize
        _print_header(cb)
        cb.row_count = 0
    elseif event === :post_iter
        if cache.k % cb.every == 0
            cb.row_count += 1
            if cb.header_every > 0 && cb.row_count > 1 && (cb.row_count - 1) % cb.header_every == 0
                _print_header(cb)
            end
            _print_row(cb, cache)
        end
    elseif event === :terminate
        cb.footer || return nothing
        println(cb.io, "Terminated: :", cache.retcode,
                "  iters=", cache.k,
                "  ‖F‖=", _fmt_sci(cache.resid),
                "  evals=", cache.n_evals,
                "  elapsed=", _fmt_dec(time() - cache.t_start), "s")
    end
    return nothing
end

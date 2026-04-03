import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Wasserstein import (
    compute_eplsilon,
    extract_96_elements,
    find_optimal_rho,
    generate_distribution_in_ball,
    wasserstein_ball,
    wasserstein_distance,
)


def load_price_income(path='price_income.csv'):
    return pd.read_csv(path)


def prepare_market_inputs(demand, ration, df, scale_factor=1.0):
    demand_array = np.asarray(demand, dtype=float).reshape(-1)
    total_demand = float(np.sum(demand_array))

    w_portion = df['W'].to_numpy(dtype=float)
    pv_portion = df['PV'].to_numpy(dtype=float)

    w_denominator = max(float(np.sum(w_portion)), 1e-9)
    pv_denominator = max(float(np.sum(pv_portion)), 1e-9)

    p_w_pre = scale_factor * float(ration) * w_portion / w_denominator * total_demand
    p_pv_pre = scale_factor * float(ration) * pv_portion / pv_denominator * total_demand
    residual_load = np.maximum(demand_array - p_w_pre - p_pv_pre, 0.0)

    return {
        'demand': demand_array,
        'P_W_pre': np.asarray(p_w_pre, dtype=float),
        'P_PV_pre': np.asarray(p_pv_pre, dtype=float),
        'residual_load': np.asarray(residual_load, dtype=float),
        'total_demand': total_demand,
    }


def generate_wasserstein_scenario(daylight_mask=None, seed=42, r=1000, partial=0.95):
    np.random.seed(seed)
    fr_samples = np.random.normal(loc=0, scale=0.5, size=1000)
    fr_center = fr_samples - np.mean(fr_samples)
    fr_center_mean = float(np.mean(fr_center ** 2))

    optimal_rho, k_value = find_optimal_rho(fr_center_mean)
    epsilon = compute_eplsilon(k_value, partial, r)

    in_ball = wasserstein_ball(fr_samples, epsilon)
    f1_samples = generate_distribution_in_ball(fr_samples, epsilon)
    w_dist_f1 = wasserstein_distance(fr_samples.flatten(), f1_samples.flatten())

    delta_w = np.asarray(extract_96_elements(f1_samples), dtype=float)
    delta_pv = np.asarray(extract_96_elements(f1_samples), dtype=float)

    if daylight_mask is not None:
        daylight_mask = np.asarray(daylight_mask, dtype=bool).reshape(-1)
        delta_pv[~daylight_mask] = 0.0

    delta_pv[:28] = 0.0
    delta_pv[89:96] = 0.0

    return {
        'Fr_samples': fr_samples,
        'epsilon': float(epsilon),
        'in_ball': in_ball,
        'F1_samples': f1_samples,
        'w_dist_F1': float(w_dist_f1),
        'delta_W_0': delta_w,
        'delta_PV_0': delta_pv,
        'optimal_rho': float(optimal_rho),
    }


def quarter_to_time(idx):
    hour = idx // 4
    minute = (idx % 4) * 15
    return f'{hour:02d}:{minute:02d}'


def build_time_labels(n_periods):
    return [quarter_to_time(i) for i in range(n_periods)]


def get_time_ticks(n_periods=96, step=8):
    ticks = list(range(0, n_periods, step))
    if ticks[-1] != n_periods - 1:
        ticks.append(n_periods - 1)
    labels = [quarter_to_time(idx) for idx in ticks]
    return ticks, labels


def _contiguous_window(series, target_idx, mode, active_indices):
    if target_idx is None:
        return None

    series = np.asarray(series, dtype=float)
    active_set = set(int(idx) for idx in np.asarray(active_indices, dtype=int).tolist())
    baseline = float(series[target_idx])

    if mode == 'min':
        threshold = baseline * 1.08 if baseline > 0 else baseline + 5.0
        compare = lambda value: value <= threshold
    else:
        threshold = baseline * 0.92 if baseline > 0 else baseline - 5.0
        compare = lambda value: value >= threshold

    left = target_idx
    while left - 1 in active_set and compare(float(series[left - 1])):
        left -= 1

    right = target_idx
    while right + 1 in active_set and compare(float(series[right + 1])):
        right += 1

    return (left, right)


def summarize_extrema(series, active_mask=None):
    values = np.asarray(series, dtype=float).reshape(-1)
    if active_mask is None:
        active_indices = np.arange(values.size)
    else:
        active_indices = np.where(np.asarray(active_mask, dtype=bool).reshape(-1))[0]

    if active_indices.size == 0:
        return {
            'active_indices': active_indices,
            'global_min': None,
            'global_max': None,
            'local_maxima': [],
        }

    active_values = values[active_indices]
    min_idx = int(active_indices[int(np.argmin(active_values))])
    max_idx = int(active_indices[int(np.argmax(active_values))])

    local_maxima = []
    for position, idx in enumerate(active_indices):
        left_value = active_values[position - 1] if position > 0 else -np.inf
        right_value = active_values[position + 1] if position < active_indices.size - 1 else -np.inf
        current_value = active_values[position]
        if current_value >= left_value and current_value >= right_value:
            local_maxima.append(int(idx))

    return {
        'active_indices': active_indices,
        'global_min': {
            'idx': min_idx,
            'value': float(values[min_idx]),
            'window': _contiguous_window(values, min_idx, 'min', active_indices),
        },
        'global_max': {
            'idx': max_idx,
            'value': float(values[max_idx]),
            'window': _contiguous_window(values, max_idx, 'max', active_indices),
        },
        'local_maxima': local_maxima,
    }


def _window_label(window):
    if window is None:
        return ''
    start_idx, end_idx = window
    start_label = quarter_to_time(start_idx)
    end_label = quarter_to_time(end_idx)
    if start_idx == end_idx:
        return start_label
    return f'{start_label}-{end_label}'


def build_market_diagnostics(
    label,
    demand,
    df,
    p_w_pre,
    p_pv_pre,
    p_w_act,
    p_pv_act,
    bids_w,
    bids_pv,
    p_setting_w=None,
    p_setting_pv=None,
):
    demand = np.asarray(demand, dtype=float).reshape(-1)
    p_w_pre = np.asarray(p_w_pre, dtype=float).reshape(-1)
    p_pv_pre = np.asarray(p_pv_pre, dtype=float).reshape(-1)
    p_w_act = np.asarray(p_w_act, dtype=float).reshape(-1)
    p_pv_act = np.asarray(p_pv_act, dtype=float).reshape(-1)
    bids_w = np.asarray(bids_w, dtype=float).reshape(-1)
    bids_pv = np.asarray(bids_pv, dtype=float).reshape(-1)
    p_setting_w = None if p_setting_w is None else np.asarray(p_setting_w, dtype=float).reshape(-1)
    p_setting_pv = None if p_setting_pv is None else np.asarray(p_setting_pv, dtype=float).reshape(-1)

    residual_load = np.maximum(demand - p_w_act - p_pv_act, 0.0)
    time_labels = build_time_labels(len(demand))
    wind_active = p_w_act > 1e-6
    solar_active = p_pv_act > 1e-6

    wind_extrema = summarize_extrema(bids_w, active_mask=wind_active)
    solar_extrema = summarize_extrema(bids_pv, active_mask=solar_active)

    night_candidates = []
    for idx in wind_extrema['local_maxima']:
        if idx <= 24:
            night_candidates.append(idx)

    if night_candidates:
        wind_night_idx = max(night_candidates, key=lambda idx: bids_w[idx])
    else:
        fallback_window = np.arange(min(25, len(bids_w)))
        wind_night_idx = int(fallback_window[int(np.argmax(bids_w[fallback_window]))])

    reference_w = df['W_Ave'].to_numpy(dtype=float) if 'W_Ave' in df.columns else np.zeros_like(bids_w)
    reference_pv = df['PV_Ave'].to_numpy(dtype=float) if 'PV_Ave' in df.columns else np.zeros_like(bids_pv)

    diagnostics = {
        'label': label,
        'time_labels': time_labels,
        'demand': demand,
        'residual_load': residual_load,
        'pre': {
            'wind': p_w_pre,
            'solar': p_pv_pre,
        },
        'wind': {
            'bids': bids_w,
            'power': p_w_act,
            'reference_bid': reference_w,
            'clear_price': p_setting_w,
            'extrema': wind_extrema,
            'night_peak_idx': wind_night_idx,
        },
        'solar': {
            'bids': bids_pv,
            'power': p_pv_act,
            'reference_bid': reference_pv,
            'clear_price': p_setting_pv,
            'extrema': solar_extrema,
        },
    }

    for resource_key in ('wind', 'solar'):
        resource = diagnostics[resource_key]
        for point_name in ('global_min', 'global_max'):
            point = resource['extrema'][point_name]
            if point is None:
                continue
            idx = point['idx']
            point['time'] = quarter_to_time(idx)
            point['window_label'] = _window_label(point['window'])
            point['power'] = float(resource['power'][idx])
            point['residual_load'] = float(residual_load[idx])
            point['reference_bid'] = float(resource['reference_bid'][idx])
            if resource['clear_price'] is not None:
                point['clear_price'] = float(resource['clear_price'][idx])

    wind_night_idx = diagnostics['wind']['night_peak_idx']
    diagnostics['wind']['night_peak'] = {
        'idx': wind_night_idx,
        'time': quarter_to_time(wind_night_idx),
        'window_label': _window_label(_contiguous_window(bids_w, wind_night_idx, 'max', wind_extrema['active_indices'])),
        'value': float(bids_w[wind_night_idx]),
        'power': float(p_w_act[wind_night_idx]),
        'residual_load': float(residual_load[wind_night_idx]),
        'reference_bid': float(reference_w[wind_night_idx]),
    }
    if p_setting_w is not None:
        diagnostics['wind']['night_peak']['clear_price'] = float(p_setting_w[wind_night_idx])

    return diagnostics


def _annotate_point(ax, x_idx, y_value, label, color):
    ax.scatter([x_idx], [y_value], color=color, s=55, zorder=4)
    ax.annotate(
        label,
        xy=(x_idx, y_value),
        xytext=(10, 14),
        textcoords='offset points',
        fontsize=9,
        color=color,
        arrowprops={'arrowstyle': '->', 'color': color, 'lw': 1.0},
    )


def _configure_time_axis(ax, n_periods):
    ticks, labels = get_time_ticks(n_periods=n_periods, step=8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45)


def plot_bid_panels(legacy_diag, corrected_diag):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)
    panels = [
        (axes[0, 0], legacy_diag, 'wind', 'W Bid Price - 现图口径', '#1f77b4'),
        (axes[0, 1], corrected_diag, 'wind', 'W Bid Price - 统一口径', '#1f77b4'),
        (axes[1, 0], legacy_diag, 'solar', 'PV Bid Price - 现图口径', '#ff7f0e'),
        (axes[1, 1], corrected_diag, 'solar', 'PV Bid Price - 统一口径', '#ff7f0e'),
    ]

    for ax, diag, resource_key, title, color in panels:
        bids = diag[resource_key]['bids']
        ax.plot(np.arange(len(bids)), bids, marker='o', linestyle='-', color=color, linewidth=1.8, markersize=4)
        _configure_time_axis(ax, len(bids))
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Bid Price')
        ax.grid(True, linestyle='--', alpha=0.4)

        if resource_key == 'solar':
            point = diag['solar']['extrema']['global_min']
            _annotate_point(
                ax,
                point['idx'],
                point['value'],
                f"最低点 {point['window_label']}\n{point['value']:.1f}",
                '#d62728',
            )
        else:
            night_peak = diag['wind']['night_peak']
            global_max = diag['wind']['extrema']['global_max']
            _annotate_point(
                ax,
                night_peak['idx'],
                night_peak['value'],
                f"夜间局部高点 {night_peak['time']}\n{night_peak['value']:.1f}",
                '#2ca02c',
            )
            _annotate_point(
                ax,
                global_max['idx'],
                global_max['value'],
                f"全日最高 {global_max['time']}\n{global_max['value']:.1f}",
                '#d62728',
            )

    plt.tight_layout()
    return fig


def plot_residual_vs_bid(legacy_diag, corrected_diag):
    fig, axes = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
    resources = [
        (axes[0], 'wind', '风电报价 vs 残余负荷'),
        (axes[1], 'solar', '光伏报价 vs 残余负荷'),
    ]

    for ax, resource_key, title in resources:
        legacy_bids = legacy_diag[resource_key]['bids']
        corrected_bids = corrected_diag[resource_key]['bids']
        legacy_residual = legacy_diag['residual_load']
        corrected_residual = corrected_diag['residual_load']
        x_axis = np.arange(len(legacy_bids))

        ax.plot(x_axis, legacy_bids, color='#7f7f7f', linestyle='--', linewidth=1.8, label='旧口径报价')
        ax.plot(x_axis, corrected_bids, color='#1f77b4' if resource_key == 'wind' else '#ff7f0e', linewidth=2.0, label='统一口径报价')
        ax2 = ax.twinx()
        ax2.plot(x_axis, legacy_residual, color='#bcbd22', linestyle=':', linewidth=1.4, label='旧口径残余负荷')
        ax2.plot(x_axis, corrected_residual, color='#9467bd', linestyle='-.', linewidth=1.4, label='统一口径残余负荷')

        _configure_time_axis(ax, len(x_axis))
        ax.set_title(title)
        ax.set_ylabel('Bid Price')
        ax2.set_ylabel('Residual Load')
        ax.grid(True, linestyle='--', alpha=0.35)

        handles_left, labels_left = ax.get_legend_handles_labels()
        handles_right, labels_right = ax2.get_legend_handles_labels()
        ax.legend(handles_left + handles_right, labels_left + labels_right, loc='upper right')

    axes[-1].set_xlabel('Time')
    plt.tight_layout()
    return fig


def _format_point_line(label, point):
    clear_price = point.get('clear_price')
    clear_part = '' if clear_price is None else f"，清算价 {clear_price:.2f}"
    return (
        f"{label}：{point['window_label']}，报价 {point['value']:.2f}，"
        f"出力 {point['power']:.2f}，残余负荷 {point['residual_load']:.2f}"
        f"{clear_part}。"
    )


def _wind_peak_line(prefix, point):
    clear_price = point.get('clear_price')
    clear_part = '' if clear_price is None else f"，清算价 {clear_price:.2f}"
    return (
        f"{prefix}：{point['time']}，报价 {point['value']:.2f}，"
        f"出力 {point['power']:.2f}，残余负荷 {point['residual_load']:.2f}"
        f"{clear_part}。"
    )


def build_analysis_text(legacy_diag, corrected_diag):
    legacy_pv = legacy_diag['solar']['extrema']['global_min']
    corrected_pv = corrected_diag['solar']['extrema']['global_min']
    legacy_wind_night = legacy_diag['wind']['night_peak']
    corrected_wind_night = corrected_diag['wind']['night_peak']
    legacy_wind_max = legacy_diag['wind']['extrema']['global_max']
    corrected_wind_max = corrected_diag['wind']['extrema']['global_max']
    corrected_pv_max_power_idx = int(np.argmax(corrected_diag['solar']['power']))
    corrected_pv_max_power = float(corrected_diag['solar']['power'][corrected_pv_max_power_idx])
    corrected_wind_night_ratio = corrected_wind_night['value'] / max(corrected_wind_max['value'], 1e-9)

    lines = [
        "现图解释",
        "当前图先按旧脚本口径理解：总脚本使用 R=80，而风光子模型又额外乘了 0.01，等效是旧版 80% 口径；这部分解释说明“图为什么长这样”，不把它直接当成统一口径下的市场规律。",
        _format_point_line("光伏最低报价", legacy_pv),
        "1. 日照与发电量关系：现图里光伏价格低点没有落在正午，而是提前到上午，是因为日照和出力不是同步线性决定报价。上午 09:30-10:15 已经进入快速爬坡区，边际增量最强，市场最容易感受到“光伏供给突然变多”的压价效应。",
        "2. 供需动态：当光伏从清晨零出力转向快速放量时，系统残余负荷会先被明显压低，价格低谷往往先出现在爬坡中段，而不是等到 12:00-13:00 的绝对峰值时刻才出现。",
        "3. 调度策略影响：旧图更受历史锚点驱动，price_income.csv 里的 PV_Ave 在 09:30-10:00 本身就偏低，模型又对高光伏出力给予更强的压价权重，因此低谷被提前放大。",
        "",
        _wind_peak_line("风电夜间局部高点", legacy_wind_night),
        _wind_peak_line("风电全日最高点", legacy_wind_max),
        "1. 夜间风力资源特点：夜里边界层更稳定、部分时段风速回升，风电供给相对平稳；同时光伏退出，风电从“与光伏竞争”变成“夜间主要可再生电源”，报价更容易形成局部高点。",
        "2. 夜间需求曲线：夜间总负荷通常较低，但因为光伏为零，残余负荷并不会像白天那样被显著压扁，所以风电价格可以保持在一个偏高的平台。",
        "3. 电价形成与消纳：在旧图里，夜间确实出现了局部高点，但全日最高点并不在半夜，而在白天 10:15 左右，说明当前图既有夜间资源特征，也强烈受历史 W_Ave 锚点和模型平滑约束影响。",
        "",
        "修正后解释",
        "下面按统一口径解释：ration=0.6 直接表示 60%，风光预测出力不再额外乘 0.01，因此这部分更接近计划要求的正式分析基线。",
        _format_point_line("光伏最低报价", corrected_pv),
        f"中午最大光伏出力：{quarter_to_time(corrected_pv_max_power_idx)}，出力 {corrected_pv_max_power:.2f}。",
        "1. 日照与发电量关系：统一口径下，光伏出力峰值仍在中午前后，但最低报价提前到上午 09:30-10:15 附近，说明真正触发压价的是“出力快速爬坡 + 边际供给突然增加”，而不一定是绝对出力最大时刻。",
        "2. 供需动态：上午 10 点左右，光伏已经释放了大部分压价能力，残余负荷明显下降；到了正午，虽然出力更高，但市场已经提前完成了大部分供需再平衡，价格不一定继续下探。",
        "3. 调度策略影响：当前模型通过残余负荷和峰时标记近似表示调度压力。现实市场中，这可以对应午前备用充足、午后爬坡预留和可能的限发预期，因此低谷常常略早于太阳正午。",
        "",
        _wind_peak_line("风电夜间局部高点", corrected_wind_night),
        _wind_peak_line("风电全日最高点", corrected_wind_max),
    ]

    if corrected_wind_night['idx'] != corrected_wind_max['idx']:
        lines.append(
            f"结论判断：夜间局部峰值高，但全日最高点不在半夜。夜间局部高点约为全日最高点的 {corrected_wind_night_ratio:.0%}，说明夜间风资源确实支撑了较高报价，但白天其他约束和历史价差信号更强。"
        )
    else:
        lines.append("结论判断：统一口径下，风电全日最高点就出现在夜间，夜间风资源和零光伏竞争共同抬升了风电报价。")

    lines.extend([
        "1. 夜间风力资源特点：夜里风速回升会提高风电可用功率，而光伏为零后，风电成为夜间最主要的可再生电源，局部价格自然容易抬高。",
        "2. 夜间需求曲线：夜间总需求较白天低，但夜间残余负荷并不一定最低，因为零光伏意味着系统仍要依赖风电、火电和储能共同出清。",
        "3. 可再生能源消纳与定价：风电报价不是只看电量，还看竞争结构。夜间少了光伏压价，风电可以维持局部高位；但如果白天某些时段历史报价锚点更高、系统约束更紧，白天仍可能出现全日最高点。",
    ])

    return '\n'.join(lines)

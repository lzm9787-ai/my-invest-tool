import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ==========================================
# 1. 页面基础设置
# ==========================================
st.set_page_config(page_title="定投策略回测工具", layout="wide")
st.title("📈 智能定投策略回测工具")
st.markdown("这是基于历史数据的定投回测演示。您可以调整侧边栏的参数，查看不同策略下的收益表现。")

# ==========================================
# 2. 解决云端中文显示问题 (关键步骤)
# ==========================================
def configure_plots():
    plt.rcParams['axes.unicode_minus'] = False
    # 尝试多种常见字体，适配不同系统（Windows/Linux/Mac）
    fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'sans-serif']
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            break
        except:
            continue
configure_plots()

# ==========================================
# 3. 数据加载函数 (带缓存，提升速度)
# ==========================================
@st.cache_data
def load_data(uploaded_file):
    try:
        # 使用你原本的逻辑读取复杂表头
        df_raw = pd.read_csv(uploaded_file, header=None)
        
        names_row = 3
        start_data_row = 5
        
        # 处理收盘价数据
        close_names = df_raw.iloc[names_row, 0:37].values
        close_names[0] = 'date'
        
        df_close = df_raw.iloc[start_data_row:, 0:37].copy()
        df_close.columns = close_names
        df_close['date'] = pd.to_datetime(df_close['date'], errors='coerce')
        df_close.set_index('date', inplace=True)
        
        # 确保全部转为数值型
        for col in df_close.columns:
            df_close[col] = pd.to_numeric(df_close[col], errors='coerce')
            
        return df_close
    except Exception as e:
        st.error(f"数据解析失败: {e}")
        return None

# ==========================================
# 4. 回测逻辑核心 (从你的类中提取并简化)
# ==========================================
def run_backtest(df, target_index, tp_configs, start_date, end_date):
    # 筛选时间
    mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
    data = df.loc[mask, target_index].dropna()
    
    if data.empty:
        return None, "该时间段无数据"

    # 初始化变量
    cash = 0
    share = 0
    total_invest = 0
    base_invest = 1000 # 假设每次定投1000元
    
    history = []
    last_tp_idx = -1
    
    for date, price in data.items():
        # 1. 买入 (定投)
        share += base_invest / price
        total_invest += base_invest
        
        # 2. 计算当前状态
        current_value = share * price
        current_return = (current_value - total_invest) / total_invest if total_invest > 0 else 0
        
        action = None
        
        # 3. 止盈检查
        # 如果收益率为负，重置止盈等级（根据你的逻辑调整）
        if current_return < 0:
            last_tp_idx = -1
            
        for idx, conf in enumerate(tp_configs):
            # 只有达到更高一级，且满足收益率要求才卖出
            if idx > last_tp_idx and current_return >= conf['return']:
                sell_ratio = conf['sell_pct']
                sell_share = share * sell_ratio
                
                cash += sell_share * price
                share -= sell_share
                
                last_tp_idx = idx
                action = f"止盈 L{idx+1}"
                break # 同一天只触发一次
        
        total_asset = cash + (share * price)
        nav = total_asset # 这里的nav其实是总资产
        
        history.append({
            'date': date,
            'price': price,
            'nav': nav,
            'invest': total_invest,
            'return': (nav - total_invest) / total_invest,
            'action': action
        })
        
    return pd.DataFrame(history), None

# ==========================================
# 5. 侧边栏：用户控制区
# ==========================================
st.sidebar.header("⚙️ 参数设置")

# 文件上传
uploaded_file = st.sidebar.file_uploader("上传数据文件 (CSV)", type=['csv'])
# 如果没有上传，尝试读取本地默认文件（方便你本地调试）
if not uploaded_file:
    try:
        default_csv = "申万行业及宽基指数.csv"
        # 只是为了演示，实际部署时建议必须上传或将文件打包
        import os
        if os.path.exists(default_csv):
            uploaded_file = default_csv
            st.sidebar.info(f"使用默认数据: {default_csv}")
    except:
        pass

if uploaded_file:
    df_close = load_data(uploaded_file)
    
    if df_close is not None:
        # 指数选择
        indices = list(df_close.columns)
        default_idx = indices.index('创业板指') if '创业板指' in indices else 0
        target_index = st.sidebar.selectbox("选择回测指数", indices, index=default_idx)
        
        # 时间选择
        min_date = df_close.index.min().date()
        max_date = df_close.index.max().date()
        
        col1, col2 = st.sidebar.columns(2)
        start_date = col1.date_input("开始日期", min_date)
        end_date = col2.date_input("结束日期", max_date)
        
        # 止盈策略配置
        st.sidebar.subheader("💰 止盈策略配置")
        
        tp_configs = []
        # Level 1
        with st.sidebar.expander("第一级止盈", expanded=True):
            r1 = st.number_input("触发收益率 (%)", value=30.0, key="r1") / 100
            s1 = st.number_input("卖出仓位 (%)", value=20.0, key="s1") / 100
            tp_configs.append({'return': r1, 'sell_pct': s1})
            
        # Level 2
        with st.sidebar.expander("第二级止盈", expanded=False):
            r2 = st.number_input("触发收益率 (%)", value=50.0, key="r2") / 100
            s2 = st.number_input("卖出仓位 (%)", value=30.0, key="s2") / 100
            tp_configs.append({'return': r2, 'sell_pct': s2})
            
        # 运行按钮
        if st.button("开始回测", type="primary"):
            res, msg = run_backtest(df_close, target_index, tp_configs, start_date, end_date)
            
            if msg:
                st.error(msg)
            else:
                # ==========================================
                # 6. 结果展示区
                # ==========================================
                final = res.iloc[-1]
                
                # 关键指标卡片
                k1, k2, k3 = st.columns(3)
                k1.metric("最终总资产", f"{final['nav']:,.0f} 元")
                k2.metric("累计投入本金", f"{final['invest']:,.0f} 元")
                ret_pct = final['return'] * 100
                k3.metric("总收益率", f"{ret_pct:.2f}%", delta=f"{ret_pct:.2f}%")
                
                # 绘图
                st.subheader("📊 净值走势图")
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                # 绘制指数 (右轴，灰色背景)
                ax2 = ax1.twinx()
                ax2.plot(res['date'], res['price'], color='gray', alpha=0.3, label='指数价格')
                ax2.set_ylabel('指数点位', color='gray')
                
                # 绘制净值 (左轴，红色实线)
                ax1.plot(res['date'], res['nav'], color='#ff4b4b', linewidth=2, label='账户资产')
                ax1.set_ylabel('账户资产 (元)', color='#ff4b4b')
                
                # 标记止盈点
                sells = res[res['action'].notna()]
                if not sells.empty:
                    ax1.scatter(sells['date'], sells['nav'], color='green', marker='v', s=100, label='止盈卖出', zorder=5)
                
                # 图例和样式
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                ax1.grid(True, alpha=0.3)
                ax1.set_title(f"{target_index} 定投回测结果 ({start_date} 至 {end_date})")
                
                st.pyplot(fig)
                
                # 详细数据表
                with st.expander("查看详细交易流水"):
                    st.dataframe(res)

    else:
        st.warning("数据加载未完成，请检查文件格式。")
else:
    st.info("👈 请在左侧上传 CSV 文件开始回测。")
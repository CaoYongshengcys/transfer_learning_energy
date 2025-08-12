import numpy as np
import tensorflow as tf
# 引入 clone_model 和 backend
from tensorflow.keras.models import Sequential, Model, clone_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import backend as K # 引入 backend
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, jsonify, request
import logging

# 禁用TensorFlow的详细日志，使输出更干净
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -------------------------------------
# 1. Flask应用和全局变量初始化
# -------------------------------------
app = Flask(__name__)

# 全局变量，用于在不同请求间保存模型和scaler
base_model = None
scalers = {
    'source': None,
    'target1': None,
    'target2': None
}

# -------------------------------------
# 2. 数据模拟函数 (无变动)
# -------------------------------------
def generate_energy_data(region_type, num_samples=365 * 24):
    # 将 np.random.seed(42) 移动到函数外部，确保随机性在不同调用间延续
    # 如果放在函数内部，每次调用生成的数据模式都是完全一样的
    hours = np.arange(num_samples) % 24
    days = np.arange(num_samples) // 24
    is_weekend = ((days % 7) >= 5).astype(int)
    base_cycle = np.sin(hours / 24.0 * 2 * np.pi - np.pi/2) + 1.2
    
    if region_type == 'source':
        demand = base_cycle * 100 + is_weekend * -15
        noise = np.random.normal(0, 5, num_samples)
    elif region_type == 'target1':
        evening_peak = np.sin((hours - 8) / 12.0 * np.pi) * 25
        evening_peak[evening_peak < 0] = 0
        demand = base_cycle * 90 + evening_peak + is_weekend * -10
        noise = np.random.normal(0, 7, num_samples)
    elif region_type == 'target2':
        midday_dip = np.sin((hours - 6) / 12.0 * np.pi) * -20
        midday_dip[midday_dip < 0] = 0
        demand = base_cycle * 110 + midday_dip + is_weekend * -25
        noise = np.random.normal(0, 6, num_samples)
    else:
        raise ValueError("未知的地区类型")
        
    demand += noise
    demand[demand < 0] = 0
    X = np.vstack([hours, is_weekend]).T
    y = demand.reshape(-1, 1)
    return X, y

# -------------------------------------
# 3. 模型构建函数 (核心修正部分)
# -------------------------------------
def build_base_model(input_shape):
    """构建基础模型，并在开始前清理会话。"""
    # 【关键修正】清理会话，确保计算图干净
    K.clear_session()
    
    model = Sequential([
        Dense(64, activation='relu', input_shape=input_shape, name='base_dense_1'),
        Dense(32, activation='relu', name='base_dense_2'),
        Dense(1, name='output_layer')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_transfer_model(base_model_to_transfer):
    """创建迁移模型，不清理会话以保持基础模型可用。"""
    # 【关键修正】不清理会话，保持基础模型可用
    # K.clear_session()  # 注释掉这行，避免销毁基础模型
    
    # 直接克隆基础模型并设置权重
    transfer_model = clone_model(base_model_to_transfer)
    transfer_model.set_weights(base_model_to_transfer.get_weights())

    # 冻结基础模型的层
    for layer in transfer_model.layers[:-1]:  # 除了最后一层
        layer.trainable = False
    
    # 确保最后一层可训练
    transfer_model.layers[-1].trainable = True
    
    # 重新编译模型
    transfer_model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 添加调试信息
    print(f"迁移模型层数: {len(transfer_model.layers)}")
    for i, layer in enumerate(transfer_model.layers):
        print(f"第{i+1}层: {layer.name}, 可训练: {layer.trainable}")
    
    return transfer_model

# -------------------------------------
# 4. Flask Web 路由 (无变动)
# -------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_model', methods=['POST'])
def run_model():
    global base_model, scalers
    # 每次请求都重置随机种子，以保证每次演示的结果都一样
    np.random.seed(42)
    region = request.json['region']
    
    try:
        if region == 'source':
            X, y = generate_energy_data('source', num_samples=365 * 24)
            scalers['source'] = StandardScaler()
            X_scaled = scalers['source'].fit_transform(X)
            base_model = build_base_model(input_shape=[X_scaled.shape[1]])
            base_model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)
            predictions = base_model.predict(X_scaled)
            num_display_points = 7 * 24
            return jsonify({
                'status': 'success',
                'message': '基础模型(华北地区)训练完成。',
                'labels': list(range(num_display_points)),
                'actual': y[-num_display_points:].flatten().tolist(),
                'predicted': predictions[-num_display_points:].flatten().tolist()
            })

        elif region in ['target1', 'target2']:
            if base_model is None:
                return jsonify({'status': 'error', 'message': '请先训练基础模型！'})

            X, y = generate_energy_data(region, num_samples=30 * 24)
            scalers[region] = StandardScaler()
            X_scaled = scalers[region].fit_transform(X)
            X_scaled_for_base = scalers['source'].transform(X)
            base_predictions = base_model.predict(X_scaled_for_base)

            transfer_model = create_transfer_model(base_model)
            print(f"迁移模型创建完成，开始训练...")
            print(f"训练数据形状: {X_scaled.shape}, 标签形状: {y.shape}")
            
            # 训练迁移模型
            transfer_model.fit(X_scaled, y, epochs=15, batch_size=16, verbose=0)
            print(f"迁移模型训练完成，开始预测...")
            
            # 进行预测
            transfer_predictions = transfer_model.predict(X_scaled)
            
            # 调试信息：打印预测结果的形状和类型
            print(f"基础预测形状: {base_predictions.shape}, 类型: {type(base_predictions)}")
            print(f"迁移预测形状: {transfer_predictions.shape}, 类型: {type(transfer_predictions)}")
            print(f"实际值形状: {y.shape}, 类型: {type(y)}")
            
            # 检查预测结果是否为空
            if transfer_predictions is None or len(transfer_predictions) == 0:
                print("错误：迁移预测结果为空！")
            else:
                print(f"迁移预测前5个值: {transfer_predictions[:5].flatten()}")
            
            # 确保预测结果不为空且格式正确
            base_predictions_flat = base_predictions.flatten().tolist()
            transfer_predictions_flat = transfer_predictions.flatten().tolist()
            actual_values_flat = y.flatten().tolist()
            
            print(f"基础预测列表长度: {len(base_predictions_flat)}")
            print(f"迁移预测列表长度: {len(transfer_predictions_flat)}")
            print(f"实际值列表长度: {len(actual_values_flat)}")
            
            region_name = "华南地区" if region == 'target1' else "西北地区"
            result_data = {
                'status': 'success',
                'message': f'迁移学习到 {region_name} 完成。',
                'labels': list(range(len(actual_values_flat))),
                'actual': actual_values_flat,
                'predicted_base': base_predictions_flat,
                'predicted_transfer': transfer_predictions_flat
            }
            print(f"返回数据键: {list(result_data.keys())}")
            print(f"predicted_base存在: {'predicted_base' in result_data}")
            print(f"predicted_transfer存在: {'predicted_transfer' in result_data}")
            return jsonify(result_data)
        
        else:
            return jsonify({'status': 'error', 'message': '未知的地区'})

    except Exception as e:
        # 打印详细错误到后端控制台，方便调试
        logging.exception(f"处理请求时发生错误: {e}")
        return jsonify({'status': 'error', 'message': f'发生严重错误: {str(e)}'})

# -------------------------------------
# 5. 启动应用 (无变动)
# -------------------------------------
if __name__ == '__main__':
    print("启动Flask服务器，请在浏览器中打开 http://127.0.0.1:5000")
    app.run(debug=True)
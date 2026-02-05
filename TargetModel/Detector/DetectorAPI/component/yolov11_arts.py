from ultralytics import YOLO
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path

class YOLOv11Detector:
    """YOLOv11の検出結果を処理するクラス"""
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): YOLOモデルのパス
        """
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.results = None
    
    def predict(self, image_path: str, conf: float = 0.3, **kwargs) -> 'YOLODetector':
        """
        画像に対して推論を実行
        
        Args:
            image_path (str): 画像のパス
            conf (float): 信頼度の閾値
            **kwargs: その他のYOLOパラメータ
            
        Returns:
            self: メソッドチェーンのためself
        """
        self.results = self.model(image_path, conf=conf, **kwargs)
        self.image_path = image_path
        return self
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        検出結果をpandas DataFrameに変換
        
        Returns:
            pd.DataFrame: 検出結果のDataFrame
        """
        if self.results is None:
            return pd.DataFrame()
        
        data = []
        for r in self.results:
            boxes = r.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    confidence = boxes.conf[i].cpu().numpy()
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    data.append({
                        'x1': float(x1), 'y1': float(y1), 
                        'x2': float(x2), 'y2': float(y2),
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name,
                        'width': float(x2 - x1),
                        'height': float(y2 - y1),
                        'area': float((x2 - x1) * (y2 - y1))
                    })
        
        return pd.DataFrame(data)
    
    def to_list(self) -> List[Dict]:
        """
        検出結果をリストに変換
        
        Returns:
            List[Dict]: 検出結果のリスト
        """
        return self.to_dataframe().to_dict('records')
    
    def filter_by_confidence(self, min_conf: float) -> pd.DataFrame:
        """
        信頼度でフィルタリング
        
        Args:
            min_conf (float): 最小信頼度
            
        Returns:
            pd.DataFrame: フィルタリング後のDataFrame
        """
        df = self.to_dataframe()
        return df[df['confidence'] >= min_conf]
    
    def filter_by_class(self, class_names: Union[str, List[str]]) -> pd.DataFrame:
        """
        クラス名でフィルタリング
        
        Args:
            class_names: クラス名（文字列またはリスト）
            
        Returns:
            pd.DataFrame: フィルタリング後のDataFrame
        """
        df = self.to_dataframe()
        if isinstance(class_names, str):
            class_names = [class_names]
        return df[df['class_name'].isin(class_names)]
    
    def get_largest_detection(self) -> Optional[Dict]:
        """
        最大面積の検出結果を取得
        
        Returns:
            Dict: 最大面積の検出結果
        """
        df = self.to_dataframe()
        if df.empty:
            return None
        return df.loc[df['area'].idxmax()].to_dict()
    
    def get_highest_confidence(self) -> Optional[Dict]:
        """
        最高信頼度の検出結果を取得
        
        Returns:
            Dict: 最高信頼度の検出結果
        """
        df = self.to_dataframe()
        if df.empty:
            return None
        return df.loc[df['confidence'].idxmax()].to_dict()
    
    def save_results(self, output_path: str, format: str = 'csv'):
        """
        結果を保存
        
        Args:
            output_path (str): 出力パス
            format (str): 保存形式 ('csv', 'json', 'excel')
        """
        df = self.to_dataframe()
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        elif format.lower() == 'excel':
            df.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def display_results(self, show_image: bool = True):
        """
        結果を表示
        
        Args:
            show_image (bool): 画像も表示するか
        """
        if self.results is None:
            print("No results available. Run predict() first.")
            return
        
        # サマリー表示
        summary = self.get_summary()
        print("=== 検出結果サマリー ===")
        print(f"総検出数: {summary['total_detections']}")
        print(f"平均信頼度: {summary.get('avg_confidence', 0):.3f}")
        print("\nクラス別検出数:")
        for class_name, count in summary['classes'].items():
            print(f"  {class_name}: {count}")
        
        # DataFrame表示
        print("\n=== 詳細結果 ===")
        df = self.to_dataframe()
        if not df.empty:
            print(df.round(3))
        
        # 画像表示（オプション）
        if show_image and self.results:
            # annotated_img = self.results[0].plot()
            # plt.figure(figsize=(12, 8))
            # plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
            # plt.axis('off')
            # plt.title('Detection Results')
            # plt.show()
            pass
    
    def __len__(self) -> int:
        """検出数を返す"""
        df = self.to_dataframe()
        return len(df)
    
    def __repr__(self) -> str:
        """オブジェクトの文字列表現"""
        return f"YOLODetector(model='{Path(self.model_path).name}', detections={len(self)})"


# 使用例のクラス
class TrafficSignDetector(YOLOv11Detector):
    """交通標識検出専用クラス"""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        # 交通標識特有の設定があればここに
    
    def get_speed_signs(self) -> pd.DataFrame:
        """速度制限標識のみを取得"""
        df = self.to_dataframe()
        # 速度制限に関連するクラス名でフィルタ（クラス名は実際のモデルに合わせて調整）
        speed_keywords = ['speed', 'limit', '制限', 'km']
        mask = df['class_name'].str.contains('|'.join(speed_keywords), case=False, na=False)
        return df[mask]
    
    def get_warning_signs(self) -> pd.DataFrame:
        """警告標識のみを取得"""
        df = self.to_dataframe()
        # 警告に関連するクラス名でフィルタ
        warning_keywords = ['warning', 'caution', '警告', '注意']
        mask = df['class_name'].str.contains('|'.join(warning_keywords), case=False, na=False)
        return df[mask]

    

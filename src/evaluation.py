import numpy as np
import pandas as pd

class RecommendationEvaluator:
    """Evaluates the recommendation system performance"""
    
    @staticmethod
    def precision_at_k(recommended_items, relevant_items, k):
        """Compute precision@k"""
        recommended_set = set(recommended_items[:k])
        relevant_set = set(relevant_items)
        if len(recommended_set) == 0:
            return 0.0
        return len(recommended_set.intersection(relevant_set)) / len(recommended_set)
    
    @staticmethod
    def recall_at_k(recommended_items, relevant_items, k):
        """Compute recall@k"""
        recommended_set = set(recommended_items[:k])
        relevant_set = set(relevant_items)
        if len(relevant_set) == 0:
            return 1.0
        return len(recommended_set.intersection(relevant_set)) / len(relevant_set)
    
    @staticmethod
    def ndcg_at_k(recommended_items, relevant_items_with_ratings, k):
        """Compute NDCG@k (Normalized Discounted Cumulative Gain)"""
        rating_dict = dict(relevant_items_with_ratings)
        ideal_rankings = sorted(rating_dict.values(), reverse=True)[:k]
        idcg = sum((rel / np.log2(i + 2)) for i, rel in enumerate(ideal_rankings))
        dcg = 0
        for i, item_id in enumerate(recommended_items[:k]):
            if item_id in rating_dict:
                dcg += rating_dict[item_id] / np.log2(i + 2)
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    @staticmethod
    def diversity(recommended_items, item_features):
        """Compute diversity of recommended items based on their features"""
        if len(recommended_items) <= 1:
            return 0.0
        features = [item_features.get(item_id, []) for item_id in recommended_items]
        distances = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                set_i = set(features[i])
                set_j = set(features[j])
                if not set_i or not set_j:
                    continue
                jaccard_dist = 1 - len(set_i.intersection(set_j)) / len(set_i.union(set_j))
                distances.append(jaccard_dist)
        return sum(distances) / len(distances) if distances else 0.0
    
    @staticmethod
    def coverage(recommended_items_all_users, all_items):
        """Compute catalog coverage - what percentage of all items are recommended to at least one user"""
        recommended_unique = set()
        for items in recommended_items_all_users:
            recommended_unique.update(items)
        return len(recommended_unique) / len(all_items)
    
    @staticmethod
    def evaluate_historical_recommendations(recommender, test_interactions, aspirants_df, k=5, verbose=True):
        """
        Evaluate the recommender system on historical interaction data.
        
        Parameters:
            recommender: Instance of your recommendation class
            test_interactions: DataFrame containing historical interactions (with aspirant_id, mentor_id, and rating)
            aspirants_df: DataFrame containing aspirants' data
            k: Number of top recommendations to consider for evaluation
            verbose: Whether to print detailed evaluation metrics
            
        Returns:
            A dictionary with average precision, recall, and NDCG.
        """
        results = {
            'precision': [],
            'recall': [],
            'ndcg': []
        }
        
        aspirant_interactions = test_interactions.groupby('aspirant_id')
        
        for aspirant_id, interactions in aspirant_interactions:
            try:
                aspirant_data = aspirants_df[aspirants_df['aspirant_id'] == aspirant_id].iloc[0].to_dict()
            except IndexError:
                continue
            
            relevant_items = interactions[interactions['rating'] >= 4]['mentor_id'].tolist()
            if not relevant_items:
                continue
            
            relevant_with_ratings = list(zip(
                interactions['mentor_id'].tolist(),
                interactions['rating'].tolist()
            ))
            
            recommendations, _ = recommender.recommend(aspirant_data, top_k=k)
            recommended_ids = recommendations['mentor_id'].tolist()
            
            precision = RecommendationEvaluator.precision_at_k(recommended_ids, relevant_items, k)
            recall = RecommendationEvaluator.recall_at_k(recommended_ids, relevant_items, k)
            ndcg = RecommendationEvaluator.ndcg_at_k(recommended_ids, relevant_with_ratings, k)
            
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['ndcg'].append(ndcg)
        
        avg_results = {
            'avg_precision': np.mean(results['precision']) if results['precision'] else 0,
            'avg_recall': np.mean(results['recall']) if results['recall'] else 0,
            'avg_ndcg': np.mean(results['ndcg']) if results['ndcg'] else 0
        }
        
        if verbose:
            print(f"Evaluation results at k={k}:")
            print(f"Average Precision: {avg_results['avg_precision']:.4f}")
            print(f"Average Recall: {avg_results['avg_recall']:.4f}")
            print(f"Average NDCG: {avg_results['avg_ndcg']:.4f}")
        
        return avg_results
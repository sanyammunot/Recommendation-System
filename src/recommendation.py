from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class MentorRecommender:
    """Advanced recommendation engine for mentor matching"""
    
    def __init__(self, mentors_df: pd.DataFrame, interactions_df: pd.DataFrame = None):
        self.mentors_df = mentors_df
        self.interactions_df = interactions_df
        self.vectorizer = None
        self.pca = None
        self.kmeans = None
        self.features_encoded = None
        
    def preprocess_features(self) -> pd.DataFrame:
        """Process mentor features into a format suitable for recommendation"""
        df = self.mentors_df.copy()
        
        # Process list features into one-hot encoded features.
        # Updated lambda: if x is already a list, leave it; otherwise, if it's NaN, return [].
        list_features = ['strengths', 'alumni_of', 'mentoring_styles', 'languages', 'specializations']
        encoded_features = pd.DataFrame(index=df.index)
        
        for feature in list_features:
            df[feature] = df[feature].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else x))
            all_values = set()
            for values_list in df[feature]:
                all_values.update(values_list)
            for value in all_values:
                encoded_features[f"{feature}_{value}"] = df[feature].apply(lambda x: 1 if value in x else 0)
        
        # Process numerical features
        numerical_features = ['expertise_level', 'clat_score', 'rank', 'years_experience', 
                              'rating', 'reviews_count', 'hourly_rate', 'availability_hours_per_week']
        for feature in numerical_features:
            if feature in df.columns:
                df[feature] = df[feature].fillna(df[feature].median())
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        numerical_data = df[numerical_features].copy()
        scaled_data = scaler.fit_transform(numerical_data)
        scaled_df = pd.DataFrame(scaled_data, columns=numerical_features, index=df.index)
        
        self.features_encoded = pd.concat([encoded_features, scaled_df], axis=1)
        
        # Add text features using TF-IDF
        if 'bio' in df.columns:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=20)
            tfidf_matrix = self.vectorizer.fit_transform(df['bio'].fillna(''))
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                                    columns=[f'bio_term_{i}' for i in range(tfidf_matrix.shape[1])],
                                    index=df.index)
            self.features_encoded = pd.concat([self.features_encoded, tfidf_df], axis=1)
        
        # Apply PCA for dimensionality reduction
        n_components = min(10, len(self.features_encoded.columns))
        self.pca = PCA(n_components=n_components)
        pca_result = self.pca.fit_transform(self.features_encoded)
        
        # Use K-means for clustering mentors (useful for cold-start recommendations)
        n_clusters = min(5, len(df))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(pca_result)
        
        # Add cluster information to the original dataframe
        self.mentors_df['cluster'] = cluster_labels
        
        return self.features_encoded
    
    def _create_aspirant_vector(self, aspirant: dict) -> pd.DataFrame:
        """Create a feature vector for an aspirant that matches the mentor feature space"""
        aspirant_vector = pd.DataFrame(0, index=[0], columns=self.features_encoded.columns)
        
        # Map preferred subjects to mentor strengths
        for subject in aspirant.get('preferred_subjects', []):
            col_name = f"strengths_{subject}"
            if col_name in aspirant_vector.columns:
                aspirant_vector[col_name] = 1
                
        # Map target colleges to mentor alumni_of
        for college in aspirant.get('target_colleges', []):
            col_name = f"alumni_of_{college}"
            if col_name in aspirant_vector.columns:
                aspirant_vector[col_name] = 1
                
        # Map learning style to mentor mentoring_styles
        learning_style = aspirant.get('learning_style')
        if learning_style:
            col_name = f"mentoring_styles_{learning_style}"
            if col_name in aspirant_vector.columns:
                aspirant_vector[col_name] = 1
                
        # Map languages
        for language in aspirant.get('languages', []):
            col_name = f"languages_{language}"
            if col_name in aspirant_vector.columns:
                aspirant_vector[col_name] = 1
                
        # Map weaknesses to mentor specializations
        for weakness in aspirant.get('weaknesses', []):
            col_name = f"specializations_{weakness}"
            if col_name in aspirant_vector.columns:
                aspirant_vector[col_name] = 1
                
        # Map numerical features
        prep_level_map = {"Beginner": 0.2, "Intermediate": 0.5, "Advanced": 0.8}
        if 'preparation_level' in aspirant and 'expertise_level' in aspirant_vector.columns:
            aspirant_vector['expertise_level'] = prep_level_map.get(aspirant['preparation_level'], 0.5)
            
        if aspirant.get('budget_constraint') and 'hourly_rate' in aspirant_vector.columns:
            aspirant_vector['hourly_rate'] = 0.2
        elif 'hourly_rate' in aspirant_vector.columns:
            aspirant_vector['hourly_rate'] = 0.5
            
        if 'months_to_exam' in aspirant and 'availability_hours_per_week' in aspirant_vector.columns:
            if aspirant['months_to_exam'] <= 3:
                aspirant_vector['availability_hours_per_week'] = 0.8
            else:
                aspirant_vector['availability_hours_per_week'] = 0.5
        
        if 'weaknesses' in aspirant and 'Time Management' in aspirant['weaknesses']:
            if 'years_experience' in aspirant_vector.columns:
                aspirant_vector['years_experience'] = 0.8
                
        if 'rating' in aspirant_vector.columns:
            aspirant_vector['rating'] = 0.7
            
        return aspirant_vector
        
    def recommend(self, aspirant: dict, top_k: int = 5, include_explanation: bool = False) -> tuple:
        """Generate personalized mentor recommendations for an aspirant"""
        if self.features_encoded is None:
            self.preprocess_features()
            
        aspirant_vector = self._create_aspirant_vector(aspirant)
        similarity_scores = cosine_similarity(self.features_encoded, aspirant_vector)
        self.mentors_df['similarity_score'] = similarity_scores
        
        top_mentors = self.mentors_df.sort_values(by='similarity_score', ascending=False).head(top_k)
        
        if not include_explanation:
            return top_mentors, {}
            
        explanations = {}
        for _, mentor in top_mentors.iterrows():
            mentor_id = mentor['mentor_id']
            subject_matches = set(mentor['strengths']).intersection(set(aspirant.get('preferred_subjects', [])))
            college_matches = set(mentor['alumni_of']).intersection(set(aspirant.get('target_colleges', [])))
            style_match = aspirant.get('learning_style') in mentor['mentoring_styles']
            weakness_coverage = set(mentor['specializations']).intersection(set(aspirant.get('weaknesses', [])))
            
            explanations[mentor_id] = {
                'subject_matches': list(subject_matches),
                'college_matches': list(college_matches),
                'style_match': style_match,
                'weakness_coverage': list(weakness_coverage),
                'experience_years': mentor['years_experience'],
                'clat_score': mentor['clat_score'],
                'rating': mentor['rating'],
                'hourly_rate': mentor['hourly_rate']
            }
            
        return top_mentors, explanations
    
    def collaborative_filtering(self, aspirant_id: int, top_k: int = 5) -> pd.DataFrame:
        """Use historical interaction data to find similar aspirants and their preferred mentors"""
        if self.interactions_df is None:
            return pd.DataFrame()
        
        aspirant_interactions = self.interactions_df[self.interactions_df['aspirant_id'] == aspirant_id]
        if len(aspirant_interactions) == 0:
            return pd.DataFrame()
            
        liked_mentors = aspirant_interactions[aspirant_interactions['rating'] >= 4]['mentor_id'].unique()
        
        similar_aspirants = self.interactions_df[
            (self.interactions_df['mentor_id'].isin(liked_mentors)) & 
            (self.interactions_df['rating'] >= 4) &
            (self.interactions_df['aspirant_id'] != aspirant_id)
        ]['aspirant_id'].unique()
        
        recommended_mentors = self.interactions_df[
            (self.interactions_df['aspirant_id'].isin(similar_aspirants)) &
            (self.interactions_df['rating'] >= 4) &
            (~self.interactions_df['mentor_id'].isin(aspirant_interactions['mentor_id']))
        ]
        
        mentor_counts = recommended_mentors['mentor_id'].value_counts().reset_index()
        mentor_counts.columns = ['mentor_id', 'recommendation_count']
        
        top_recommendations = pd.merge(
            mentor_counts,
            self.mentors_df,
            on='mentor_id'
        ).sort_values(by='recommendation_count', ascending=False).head(top_k)
        
        return top_recommendations
    
    def hybrid_recommendation(self, aspirant: Dict[str, Any], aspirant_id: Optional[int] = None,
                             top_k: int = 5, content_weight: float = 0.7) -> Tuple[pd.DataFrame, Dict[int, Dict]]:
        """Combine content-based and collaborative filtering approaches"""
        # Get content-based recommendations
        content_rec, _ = self.recommend(aspirant, top_k=top_k*2)
        
        # Get collaborative recommendations
        collab_rec = pd.DataFrame()
        if aspirant_id is not None and self.interactions_df is not None:
            collab_rec = self.collaborative_filtering(aspirant_id, top_k=top_k*2)
        
        # If no collaborative results, return content-based
        if len(collab_rec) == 0:
            top_hybrid = content_rec.head(top_k)
            explanations = self._generate_explanations(aspirant, top_hybrid)
            return top_hybrid, explanations
        
        # Normalize scores
        content_rec['normalized_score'] = self._normalize_scores(content_rec['similarity_score'])
        collab_rec['normalized_score'] = self._normalize_scores(collab_rec['recommendation_count'])
        
        # Combine recommendations
        combined = pd.concat([
            content_rec[['mentor_id', 'normalized_score']].assign(source='content'),
            collab_rec[['mentor_id', 'normalized_score']].assign(source='collab')
        ])
        
        # Pivot and calculate hybrid score
        pivot_scores = combined.pivot_table(
            index='mentor_id', 
            columns='source', 
            values='normalized_score',
            aggfunc='first'
        ).fillna(0)
        
        pivot_scores['hybrid_score'] = (
            content_weight * pivot_scores.get('content', 0) + 
            (1 - content_weight) * pivot_scores.get('collab', 0)
        )
        
        # Merge with mentor data
        top_hybrid = pd.merge(
            pivot_scores[['hybrid_score']].reset_index(),
            self.mentors_df,
            on='mentor_id'
        ).sort_values('hybrid_score', ascending=False).head(top_k)
        
        # Generate explanations
        explanations = self._generate_explanations(aspirant, top_hybrid)
        
        return top_hybrid, explanations
    
    def _normalize_scores(self, scores: pd.Series) -> pd.Series:
        """Normalize scores to 0-1 range"""
        if scores.nunique() == 0:
            return pd.Series([0.5]*len(scores), index=scores.index)
        return (scores - scores.min()) / (scores.max() - scores.min())

    def _generate_explanations(self, aspirant: Dict[str, Any], recommendations: pd.DataFrame) -> Dict[int, Dict]:
        """Generate explanations for all recommendations"""
        explanations = {}
        for _, mentor in recommendations.iterrows():
            mentor_id = mentor['mentor_id']
            explanations[mentor_id] = {
                'subject_matches': list(set(mentor['strengths']).intersection(aspirant.get('preferred_subjects', []))),
                'college_matches': list(set(mentor['alumni_of']).intersection(aspirant.get('target_colleges', []))),
                'style_match': aspirant.get('learning_style') in mentor['mentoring_styles'],
                'weakness_coverage': list(set(mentor['specializations']).intersection(aspirant.get('weaknesses', []))),
                'experience': mentor['years_experience'],
                'rating': mentor['rating']
            }
        return explanations

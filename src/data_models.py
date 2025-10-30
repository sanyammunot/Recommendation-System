import pandas as pd
import random
from datetime import datetime

class DataSchema:
    """Contains all data schema definitions"""
    SUBJECTS = [
        "Legal Reasoning", "English", "Current Affairs", "Logical Reasoning", 
        "Quantitative Techniques", "General Knowledge", "Static GK", 
        "Legal Aptitude", "Constitutional Law", "Criminal Law"
    ]
    COLLEGES = [
        "NLSIU Bangalore", "NLU Delhi", "NALSAR Hyderabad", 
        "WBNUJS Kolkata", "NLU Jodhpur", "JGLS Sonipat", 
        "NLIU Bhopal", "GNLU Gandhinagar", "RGNUL Patiala", 
        "MNLU Mumbai"
    ]
    PREPARATION_LEVELS = ["Beginner", "Intermediate", "Advanced"]
    LEARNING_STYLES = ["Visual", "Auditory", "Reading/Writing", "Kinesthetic"]
    EXPERTISE_LEVELS = [1, 2, 3, 4, 5]
    WEAKNESSES = [
        "Time Management", "Reading Comprehension", "Critical Analysis",
        "Current Affairs Knowledge", "Legal Terminology", "Quantitative Skills",
        "Exam Anxiety", "Mock Test Performance", "English Vocabulary",
        "Legal Maxims", "Constitutional Concepts", "Logical Puzzles"
    ]
    LANGUAGES = ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Kannada", "Marathi", "Gujarati"]
    CLAT_SCORE_RANGE = (120, 150)
    YEARS_EXPERIENCE_RANGE = (1, 5)
    AVAILABILITY_HOURS_RANGE = (5, 20)

    
class DataGenerator:
    """Class to generate realistic mock data"""
    
    @staticmethod
    def generate_aspirants(n=100) -> pd.DataFrame:
        """Generate a dataset of aspirants"""
        aspirants = []
        
        for i in range(1, n + 1):
            # Choose 2-4 preferred subjects
            preferred_subjects = random.sample(DataSchema.SUBJECTS, k=random.randint(2, 4))
            
            # Choose 1-3 target colleges
            target_colleges = random.sample(DataSchema.COLLEGES, k=random.randint(1, 3))
            
            # Choose 2-4 weaknesses
            weaknesses = random.sample(DataSchema.WEAKNESSES, k=random.randint(2, 4))
            
            # Choose 1-3 spoken languages
            languages = random.sample(DataSchema.LANGUAGES, k=random.randint(1, 3))
            
            # Generate mock test scores (out of 150)
            recent_mock_scores = [
                random.randint(50, 130) for _ in range(random.randint(1, 5))
            ]
            
            aspirant = {
                "aspirant_id": i,
                "name": f"Aspirant_{i}",
                "preferred_subjects": preferred_subjects,
                "target_colleges": target_colleges,
                "preparation_level": random.choice(DataSchema.PREPARATION_LEVELS),
                "learning_style": random.choice(DataSchema.LEARNING_STYLES),
                "weaknesses": weaknesses,
                "languages": languages,
                "recent_mock_scores": recent_mock_scores,
                "avg_mock_score": round(sum(recent_mock_scores) / len(recent_mock_scores), 2) if recent_mock_scores else 0,
                "study_hours_per_week": random.randint(10, 60),
                "months_to_exam": random.randint(1, 12),
                "previous_attempts": random.randint(0, 2),
                "budget_constraint": random.choice([True, False]),
                "registration_date": datetime.now().strftime("%Y-%m-%d")
            }
            aspirants.append(aspirant)
            
        return pd.DataFrame(aspirants)
    
    @staticmethod
    def generate_mentors(n=30) -> pd.DataFrame:
        """Generate a dataset of mentors"""
        mentors = []
        
        for i in range(101, 101 + n):
            # Choose 3-5 subjects of strength
            strengths = random.sample(DataSchema.SUBJECTS, k=random.randint(3, 5))
            
            # Choose 1-2 alma maters
            alumni_of = random.sample(DataSchema.COLLEGES, k=random.randint(1, 2))
            
            # Choose 1-2 mentoring styles
            mentoring_styles = random.sample(DataSchema.LEARNING_STYLES, k=random.randint(1, 2))
            
            # Choose 1-3 languages
            languages = random.sample(DataSchema.LANGUAGES, k=random.randint(1, 3))
            
            # Generate a realistic CLAT score 
            clat_score = random.randint(*DataSchema.CLAT_SCORE_RANGE)
            
            # Generate rating based on a normal distribution (clamped between 1.0 and 5.0)
            rating = min(5.0, max(1.0, random.normalvariate(4.0, 0.5)))
            
            mentor = {
                "mentor_id": i,
                "name": f"Mentor_{i}",
                "strengths": strengths,
                "alumni_of": alumni_of,
                "expertise_level": random.choice(DataSchema.EXPERTISE_LEVELS),
                "mentoring_styles": mentoring_styles,
                "clat_score": clat_score,
                "rank": random.randint(1, 500),
                "years_experience": random.randint(*DataSchema.YEARS_EXPERIENCE_RANGE),
                "languages": languages,
                "rating": round(rating, 1),
                "reviews_count": random.randint(0, 50),
                "hourly_rate": random.randint(300, 1500),
                "availability_hours_per_week": random.randint(*DataSchema.AVAILABILITY_HOURS_RANGE),
                "specializations": random.sample(DataSchema.WEAKNESSES, k=random.randint(2, 4)),
                "verified": random.choice([True, False]),
                "bio": f"Experienced CLAT mentor with expertise in {', '.join(strengths[:2])}. Alumni of {', '.join(alumni_of)}."
            }
            mentors.append(mentor)
            
        return pd.DataFrame(mentors)
    
    @staticmethod
    def generate_interaction_data(aspirants_df, mentors_df, interaction_ratio=0.3) -> pd.DataFrame:
        """Generate historical interaction data between aspirants and mentors"""
        interactions = []
        
        # Determine how many interactions to generate
        num_interactions = int(len(aspirants_df) * len(mentors_df) * interaction_ratio)
        
        for _ in range(num_interactions):
            aspirant_id = random.choice(aspirants_df['aspirant_id'].tolist())
            mentor_id = random.choice(mentors_df['mentor_id'].tolist())
            
            # Generate a random timestamp within the last 6 months
            days_ago = random.randint(1, 180)
            timestamp = (datetime.now() - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")
            
            # Generate a random rating between 1 and 5
            rating = random.randint(1, 5)
            
            # Generate a random session duration between 30 minutes and 2 hours
            session_duration_mins = random.randint(30, 120)
            
            # Choose one of the mentor's strengths as the session topic
            mentor = mentors_df[mentors_df['mentor_id'] == mentor_id].iloc[0]
            session_topic = random.choice(mentor['strengths'])
            
            interaction = {
                "interaction_id": len(interactions) + 1,
                "aspirant_id": aspirant_id,
                "mentor_id": mentor_id,
                "timestamp": timestamp,
                "rating": rating,
                "session_duration_mins": session_duration_mins,
                "session_topic": session_topic,
                "feedback": random.choice([
                    "Very helpful session!", 
                    "Mentor explained concepts clearly", 
                    "Need to cover more practice questions", 
                    "Great strategies for time management",
                    "Would prefer more structured approach",
                    "Excellent at breaking down complex topics",
                    "Helped improve my weak areas"
                ])
            }
            interactions.append(interaction)
            
        return pd.DataFrame(interactions)

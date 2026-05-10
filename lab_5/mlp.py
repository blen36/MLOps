import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = pd.DataFrame({
    "text": [
        "The food was absolutely amazing and delicious", "Great service, highly recommend this place",
        "I loved the cozy atmosphere and fresh pasta", "Best restaurant in town, fantastic experience",
        "Really good coffee and friendly staff", "Superb dinner, everything was perfect",
        "Delicious dessert, I will definitely come back", "Awesome food, portions are very generous",
        "Excellent quality and fast delivery", "Yummy and cheap, a hidden gem",
        "The steak was cooked to perfection, so tasty", "Wonderful evening with live music and great wine",
        "Staff was very attentive and helpful throughout", "The best pizza I have ever had in my life",
        "Everything was fresh and beautifully presented", "Highly professional service and tasty dishes",
        "A truly magical place for a romantic dinner", "Fast service even though the place was crowded",
        "Great value for money, we are very happy", "Such a lovely place with a great menu selection",
        "The seafood was incredibly fresh and flavorful", "Impressive wine list and knowledgeable sommelier",
        "The atmosphere is so relaxing, a great escape", "Perfect spot for a family lunch, kids loved it",
        "Top notch quality and very creative chef", "Clean, bright and very welcoming restaurant",
        "I am obsessed with their homemade lemonade", "Authentic flavors that remind me of Italy",
        "Quick bite but high quality, will return", "The best service we have received in a long time",
        "The breakfast was hearty and delicious", "Amazing cocktails and very cool interior",
        "Highly recommend the spicy ramen, it is great", "Warm welcome and excellent hospitality",
        "The outdoor seating is beautiful in the summer", "Phenomenal flavors in every single bite",
        "Such a pleasant surprise, everything was great", "The burger was juicy and full of flavor",
        "Lovely staff made our anniversary special", "Great portions and the prices are fair",
        "Authentic sushi, very fresh and well prepared", "The terrace has a wonderful view",
        "Quick service and very polite waiters", "Simply the best dining experience recently",
        "Healthy options that actually taste good", "The bread was warm and freshly baked",
        "Incredible attention to detail in every dish", "Cozy little cafe with the best lattes",
        "The grill mix was superb and very filling", "Nice music and not too loud, perfect for talk",
        "Wonderful textures and very balanced spices", "The owner is very friendly and welcoming",
        "Top tier kitchen, everything was top notch", "Great place for a large group of friends",
        "The pancakes were fluffy and very sweet", "Elegant presentation and refined taste",
        "We felt very well taken care of by the staff", "The salad was crisp and the dressing was great",
        "I could eat here every day, so good", "Brilliant service and mouth watering food",
        "The fish was light and perfectly seasoned", "Exceptional quality for such a price",
        "Modern and stylish place with great vibes", "The soup was rich and very comforting",
        "Professional chefs and very tasty menu", "Fast delivery and the food arrived hot",
        "The local wine is a must try, excellent", "Friendly atmosphere and very clean space",
        "Outstanding meal, exceeded all expectations", "The pasta carbonara was truly authentic",
        "A great spot for brunch on the weekend", "The ribs were tender and fell off the bone",
        "Excellent vegetarian options, very creative", "Great lighting and very comfortable seats",
        "Will definitely be my new favorite spot",

        "Terrible experience, the soup was cold", "Worst waiter ever, very slow and rude",
        "Overpriced and completely tasteless food", "Awful place, dirty tables and bad music",
        "Never coming back, disgusting meal", "Disappointing, the meat was undercooked",
        "Bad service, we waited for an hour", "Horrible, do not waste your money here",
        "Not good at all, very bland flavors", "Rude manager and extremely poor quality",
        "The bread was stale and the salad was wilted", "Extremely loud music, couldn't even talk",
        "Small portions for such a high price, sad", "The place smells bad and the floor is sticky",
        "Waited 40 minutes for a menu, then they forgot us", "The chicken was dry and lacked any seasoning",
        "Found a hair in my food, absolutely gross", "Poor management and disorganized staff",
        "The interior is falling apart, very depressing", "Medicine-like taste in the cocktails, yuck",
        "Worst experience of the year, stay away", "They charged us for items we didn't order",
        "Cold atmosphere and very unfriendly welcome", "The pasta was overcooked and mushy",
        "Kitchen is very slow, expect to wait forever", "Dirty glasses and stained napkins",
        "I had high hopes but was deeply disappointed", "The dessert was frozen in the middle",
        "Terrible value, you can find better elsewhere", "Avoid this place if you value your time and health",
        "The pizza was burnt and very oily", "Waitress was dismissive and quite arrogant",
        "The table was shaking and the chairs were broken", "Watery coffee and very dry croissants",
        "They ran out of almost everything on the menu", "Very greasy food, felt sick afterwards",
        "The acoustics are terrible, way too noisy", "Total waste of time, extremely slow service",
        "The soup tasted like it came from a can", "Hidden fees on the bill, very dishonest",
        "The steak was tough as leather, couldn't eat it", "Poor hygiene, saw a cockroach near the bar",
        "The owner was shouting at the staff, very awkward", "Everything was bland and lacked salt",
        "My order was completely wrong and they didn't care", "The fries were soggy and ice cold",
        "Unprofessional behavior from the hostess", "The place is cramped and very uncomfortable",
        "The wine was sour and tasted like vinegar", "Overwhelming smell of cleaning chemicals",
        "No vegetarian options despite what the site said", "The bathroom was disgusting and had no soap",
        "Mediocre food at premium prices, not worth it", "The service was chaotic and very confused",
        "Tasteless pasta with a very watery sauce", "The lighting was too dark, couldn't see the food",
        "Waiters were ignored us for most of the night", "Dry cake that tasted like it was a week old",
        "The air conditioning was broken, so hot inside", "Lack of flavor in everything we ordered",
        "The manager was very unhelpful with our complaint", "Old furniture and very dusty decorations",
        "The fish had a very strange and fishy smell", "Disorganized kitchen, meals arrived at different times",
        "The portions have gotten smaller and prices higher", "Stale crackers and very salty dip",
        "It took ages to get the check, very annoying", "The sauce was way too spicy, couldn't taste anything",
        "Unfriendly staff made us feel unwelcome", "Cheap ingredients used in expensive dishes",
        "The tea was lukewarm and very weak", "The meat had a lot of gristle, poor cut",
        "Very disappointing brunch, nothing was fresh", "Tables are too close together, no privacy",
        "I regret coming here, a total disaster"
    ],
    "sentiment": [1]*75 + [0]*75
})

X = data["text"]
y = data["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(stop_words="english", lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(
        C=0.5,  # Умеренная регуляризация
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,  # Больше деревьев — стабильнее результат
        max_depth=12,  # Ограничиваем глубину, чтобы не переобучиться
        random_state=42
    ),

    "MLP (Neural Network)": MLPClassifier(
        hidden_layer_sizes=(30, 20),
        alpha=0.05,
        activation='relu',
        solver='adam',
        max_iter=1000,
        learning_rate_init=0.001,
        random_state=42
    )
}

predictions = {}
accuracies = {}

print("=== СРАВНЕНИЕ МОДЕЛЕЙ ===\n")
for name, model in models.items():
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    predictions[name] = y_pred
    acc = accuracy_score(y_test, y_pred)
    accuracies[name] = acc
    print(f"--- {name} ---")
    print(f"Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=['skyblue', 'lightgreen', 'salmon'])
plt.ylim(0, 1.1)
plt.title('Сравнение точности (Accuracy) моделей', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, round(yval, 2), ha='center', fontsize=12)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Матрицы ошибок (Confusion Matrices)', fontsize=16)
for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                xticklabels=['Негативный', 'Позитивный'],
                yticklabels=['Негативный', 'Позитивный'])
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Предсказанный класс')
    axes[idx].set_ylabel('Истинный класс')
plt.tight_layout()
plt.show()
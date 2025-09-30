Définir la stratégie d’élaboration d’un modèle d'apprentissage profond, concevoir ou ré-utiliser des modèles pré-entraînés (transfer learning) et entraîner des modèles afin de réaliser une analyse prédictive.
CE1 Le candidat, en complément de la démarche de type “bag-of-words”, a mis en oeuvre 3 démarches de word/sentence embedding : Word2Vec (ou Doc2Vec ou Glove ou FastText), BERT, et USE (Universal Sentence Encoder) : 

au moins deux techniques de prétraitement du texte ont été testées (ex : lemmatization, stemming…)
au moins deux méthodes d’embedding ont été testées (parmi word2vec, Glove, fasttext) pour les “modèles avancés” Tensorflow/Keras.
Les données sont préparées (input ids, attention mask) pour mettre en oeuvre une démarche d’embedding pour le “modèle BERT”
Une démarche optionnelle d’embedding via USE est mise en oeuvre
CE2 Le candidat a défini sa stratégie d’élaboration d’un modèle pour répondre à un besoin métier (par exemple : choix de conception d’un modèle ou ré-utilisation de modèles pré-entraînés).

CE3 Le candidat a identifié la ou les cibles. 

CE4 Le candidat a réalisé la séparation du jeu de données en jeu d’entraînement, jeu de validation et jeu de test. 

CE5 Le candidat s'est assuré qu'il n’y a pas de fuite d’information entre les jeux de données (entraînement, validation et test). 

CE6 Le candidat a testé plusieurs modèles d’apprentissage profond en partant du plus simple vers les plus complexes : 

dont au moins un modèle Tensorflow/Keras de base avec embedding, un modèle Tensorflow/Keras avec embedding et couche LSTM, ainsi qu’un modèle BERT.
CE7 Le candidat a mis en oeuvre des modèles à partir de modèles pré-entraînés (technique de Transfer Learning)

 

Évaluer la performance des modèles d’apprentissage profond selon différents critères (scores, temps d'entraînement, etc.) afin de choisir le modèle le plus performant pour la problématique métier.
CE1 Le candidat a choisi une métrique adaptée à la problématique métier, et sert à évaluer la performance des modèles 

CE2 Le candidat a explicité le choix de la métrique d’évaluation 

CE3 Le candidat a évalué la performance d’un modèle de référence et sert de comparaison pour évaluer la performance des modèles plus complexes 

CE4 Le candidat a calculé, hormis la métrique choisie, au moins un autre indicateur pour comparer les modèles (par exemple : le temps nécessaire pour l’entraînement du modèle) 

CE5 Le candidat a optimisé au moins un des hyperparamètres du modèle choisi (par exemple : le choix de la fonction Loss, le Batch Size, le nombre d'Epochs) 

CE6 Le candidat a présenté une synthèse comparative des différents modèles, par exemple sous forme de tableau. 

 

Définir et mettre en œuvre un pipeline d’entraînement des modèles, avec centralisation du stockage des modèles et formalisation des résultats et mesures des différentes expérimentations réalisées, afin d’industrialiser le projet de Machine Learning.
CE1 Le candidat a mis en oeuvre un pipeline d’entraînement des modèles reproductible

CE2 Le candidat a sérialisé et stocké les modèles créés dans un registre centralisé afin de pouvoir facilement les réutiliser.

CE3 Le candidat a formalisé des mesures et résultats de chaque expérimentation, afin de les analyser et de les comparer

 

Mettre en œuvre un logiciel de version de code afin d’assurer en continu l’intégration et la diffusion du modèle auprès de collaborateurs.
CE1 Le candidat a créé un dossier contenant tous les scripts du projet dans un logiciel de version de code (ex : Git) et l'a partagé (ex : Github). 

CE2 Le candidat a présenté un historique des modifications du projet qui affiche au moins trois versions distinctes, auxquelles il est possible d'accéder. 

CE3 Le candidat a tenu à jour et mis à disposition la liste des packages utilisés ainsi que leur numéro de version . 

CE4 Le candidat a rédigé un fichier introductif permettant de comprendre l'objectif du projet et le découpage des dossiers. 

CE5 Le candidat a commenté les scripts et les fonctions facilitant une réutilisation du travail par d'autres personnes et la collaboration. 

 

Concevoir et assurer un déploiement continu d'un moteur d’inférence (modèle de prédiction encapsulé dans une API) sur une plateforme Cloud afin de permettre à des applications de réaliser des prédictions via une requête à l’API.
CE1 Le candidat a défini et préparé un pipeline de déploiement continu.

CE2 Le candidat a déployé le modèle de machine learning sous forme d'API (via Flask par exemple) et cette API renvoie bien une prédiction correspondant à une demande. 

CE3 Le candidat a mis en œuvre un pipeline de déploiement continu, afin de déployer l'API sur un serveur d'une plateforme Cloud. 

CE4 Le candidat a mis en oeuvre des tests unitaires automatisés (par exemple avec pyTest)

CE5 Le candidat a réalisé l'API indépendamment de l'application qui utilise le résultat de la prédiction. 

 

Définir et mettre en œuvre une stratégie de suivi de la performance d’un modèle en production et en assurer la maintenance afin de garantir dans le temps la production de prédictions performantes.
CE1 Le candidat a défini une stratégie de suivi de la performance du modèle. Dans le cadre du projet : 

choix d’utiliser Azure Application Insight pour le suivi de traces de prédictions non conformes et de déclenchement d’alertes
CE2 Le candidat a réalisé un système de stockage d’événements relatifs aux prédictions réalisées par l’API et une gestion d’alerte en cas de dégradation significative de la performance. Dans le cadre du projet : 

mise en oeuvre sur Azure Application Insight de de traces relatives à des prédictions non conformes, paramétrage de déclenchement d’alertes et exécution des alertes envoyées par mail ou SMS
CE3 Le candidat a analysé la stabilité du modèle dans le temps et défini des actions d’amélioration de sa performance. Dans le cadre de ce projet : 

présentation dans le blog d’une démarche qui pourrait être mise en oeuvre pour l’analyse de ces statistiques et l’amélioration du modèle dans le temps

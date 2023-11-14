# Plan
### I - IA générative
### II - Cloner une voix
### III - RVC & Hugging Face



# I - IA générative

L’IA dite “générative” est une sous-branche de l’intelligence artificielle qui se concentre sur la création, via des modèles de deep-learning, de données ou de contenus inédits. L’IA générative va plutôt se concentrer sur la génération de données “artistique” (images, textes, audio..) mais aussi structurée pour recréer un dataset (données financières crédibles...). Contrairement à l’IA “classique” qui va plus essayer de rejouer des comportements humains dans la classification, la prédicition ou la résolution de problèmes. 

### I.1 Machine Learning vs Deep Learning

||Machine Learning| Deep Learning|
|-----|-----|---|
|Définition|L'apprentissage automatique est une branche de  l'intelligence artificielle (IA) qui se concentre sur l'utilisation de données et d'algorithmes pour imiter la façon dont les humains apprennent, en améliorant progressivement leur précision.|sous-ensemble du machine learning, repose essentiellement sur un réseau de neurones à trois couches ou plus. Réseaux de neurones tentent de simuler le comportement du cerveau humain en lui permettant « d’apprendre » à partir de grandes quantités de données. Même si un réseau neuronal doté d’une seule couche peut toujours faire des prédictions approximatives, des hidden layers supplémentaires peuvent aider à optimiser et à affiner la précision|
|Type de données utilisées | données structurées et étiquetées pour effectuer des prédictions, ce qui signifie que des fonctionnalités spécifiques sont définies à partir des données d'entrée du modèle et organisées en tableaux. Cela ne signifie pas nécessairement qu’il n’utilise pas de données non structurées ; cela signifie simplement que si c'est le cas, il subit généralement un prétraitement pour l'organiser dans un format structuré| élimine une partie du prétraitement des données généralement impliqué dans le ML. Ces algorithmes peuvent ingérer et traiter des données non structurées, comme du texte et des images, et automatiser l'extraction de fonctionnalités, supprimant ainsi une partie de la dépendance vis-à-vis des experts humains (par exemple : Les algorithmes d'apprentissage profond peuvent déterminer quelles caractéristiques (par exemple les oreilles) sont les plus importantes pour distinguer chaque animal d'un autre. En apprentissage automatique, cette hiérarchie de fonctionnalités est établie manuellement par un expert humain) |
|Base de données|limitée ou controlable| supérieure à 1 million de données|
|Type d'apprentissage possible|apprentissage supervisé (ensembles de données étiquetés pour catégoriser ou faire des prédictions, nécessite action humaine), apprentissage non supervisé (nécessite pas d’ensembles de données étiquetés ; it detects patterns in the data, clustering them by any distinguishing characteristics) et apprentissage par renforcement (modèle apprend à devenir plus précis pour effectuer une action dans un environnement basé sur le feedback afin de maximiser la récompense)|supervisé, non-supervisé, par renforcement|
|Algorithmes ou modèles| Régression linéaire, Régression logistique, Clustering, Arbres de décision, Forêts aléatoires|Réseaux de neurones|

Les modèles de DL et de ML en général peuvent être divisés en 2 types : 
- discriminative (classify) : par exemple, si on donne des photos de chien à un modèle, le modèle discriminant apprend la distribution de probabilité conditionnelle, ou la probabilité de y notre sortie, etant donné x notre entrée, qu'il s'agit d'un chien, et classifie comme un chien et non  chat.
- generative (generate) : apprend la distribution de probabilité conjointe, ou la probabilité de x et y, et prédit la probabilité conditionelle qu'il s'agisse d'un chien, et ensuite peut générer une image d'un chien.

==> D'où, les modèles génératifs peuvent générer de nouvelles isntances de données, tandis que les modèles discriminants font la distinction entre différents types d'instances de données.

[Qu'est-ce que le deep learning ?](https://www.ibm.com/topics/deep-learning#:~:text=Deep%20learning%20neural%20networks%2C%20or,describe%20objects%20within%20the%20data.) ; [Qu'est-ce que l'apprentissage automatique ?](https://www.ibm.com/topics/machine-learning#What+is+machine+learning%3F)

### I.2 Gen AI

Gen AI est un sous-ensemble du deep learning. Elle utilise des artifical neural neworks, peut traiter des données étiquetées et non étiquetées, utilisant des méthodes supervised, unsupervised, and semi-supervised learning.

<img width="960" alt="distinction" src="https://github.com/iciamyplant/IA_vocales/assets/57531966/f0285993-467b-4e02-a0d5-665a86f6a341">

- Le processus d'apprentissage traditionnel, classique, supervisé et non supervisé utilise le training code et les données étiquetées pour construire un modèle. En fonction du cas d'utilisation ou du problème, le modèle peut nous donner une prédiction, classer quelque chose ou regrouper quelque chose.
- Gen AI peut prendre du training code, des données étiquetées ou non étiquetées, de tous types de données, et créer un foundation model. Le foundation model peut ensuite générer du nouveau contenu (texte, code, images, audio, vidéo, etc).

Transformers : La puissance des IA génératives vient de l'utilisation de transformers. Les transformers ont crée une révolution dans le naturel langage processing en 2018. Un transformer consiste en gros en un encodeur et un décodeur. l'encodeur encode la séquence d'entrée, et la transmet au décodeur, qui apprend à décoder la représentation pour une certaine tâche. Dans les transformateurs, les hallucinations sont des mots ou des phrases générés par le modèle qui sont souvent absurdes ou grammaticalement incorrects. 

Prompt : le court morceau de texte transmis au LLM en entrée. Prompt design = process de concpetion du prompt qui générera le résultat souhaité. Qualité du prompt détermine aussi beaucoup qualité du résultat. 

Models types
- text to text. Applications : generation, classification, summariziation, translation, extraction...
- text to image.  Application : image generation, image editing.
- text to video, and text to 3D. Application : video generation, video editing, game assets
- text to task. Trained to perfom a task or action based on text. Application : automation, virtual assisatnts, software agents.

Foundation model : est un grand modèle d'IA pré-entraîné, sur une grande quantité de données, conçu pour être adapté ou affiné à un large éventail de taches en aval du modèle. les foundaiton models ont le potentiel de révolutionner de nombreux secteurs.

[Vidéo Introduction to Generative AI, Google Cloud Tech](https://www.youtube.com/watch?v=G2fqAlgmoPo)






# II - Cloner une voix

### II.1 Etape 1 : Constitution d'une base de données

Constituer une base de données avec des enregistrements vocaux de la voix qu'on veut synthétiser. 

Là en général on a le signal audio, et on aimerait bien changer la représentation pour qu’on ait quelque chose de plus exploitable comme une image. Et le plus souvent on utilise un spectrogramme (en gros on passe dans le domaine fréquentiel, transformation de Fourier, spectre, mais par intervalle de temps et pas sur tout le signal). On a une représentation visuelle du son sur lequel on peut appliquer nos algos.

![Capture d’écran 2023-11-14 à 08 50 28](https://github.com/iciamyplant/IA_vocales/assets/57531966/c259a282-a6b5-4a0d-b765-6c2898e198ea)

Le son est une onde dont la fréquence détermine la hauteur tonale et l'amplitude le volume. Comme il existe une infinité de sons, il existe une infinité d'ondes. 



### II.2 Etape 2 : Encodage de la voix

Calculer par Deep Learning une manière d’encoder les informations d’une voix. Il y a plein de manières différentes d'encoder les infos du speaker. 

Exemple : avec une architecture encoder/décodeur. Une manière de faire est d’utiliser une architecture d’encoder-décodeur, en entrée on met la représentation du signal, on réduit la dimension avec des couches de réseaux de neurones, jusqu’à arriver à notre ”espace latent”, et derrière on refait le chemin inverse en essayant de reconstruire au mieux notre signal de base, en partant de l’espace latent. Et ensuite entraîner. Si on fait ça plein de fois, avec plein de données, la partie encoder va nous permettre d’optimiser la manière par laquelle on fait différentes opérations sur le signal de base pour arriver à la représentation latente parfaite. 

==> c’est simplifié

![Capture d’écran 2023-11-14 à 08 56 00](https://github.com/iciamyplant/IA_vocales/assets/57531966/f822fb5f-b00f-4d7a-8fe3-13b35dc9c54d)



### II.3 Etape 3 : Synthèse de la voix

Avec cet encodage de la voix, on peut l’utiliser pour ”styliser” : 
- n'importe quel signal qu'on transforme dans la voix qu'on veut
- du texte qu'on transforme en speech dans la voix qu'on veut

Pour synthétiser la voix, plein de manière de faire une fois encore. Par exemple, en utilisant un modèle de synthèse de texte générale (type Tacotron, WaveNet...) et on va rajouter la condition qui est contenue dans les informations encodées du speakers (en vert schéma).

==> A partir de là on a une représentation générale de la voix avec toutes les informations qu’il nous faut : 
- signal/texte à transformer (en bleu)
- informations sur le speaker (en vert)
- ==> Tout ça nous permet de régénérer le spectrogramme et ainsi le message dans la bonne voix.

![Capture d’écran 2023-11-14 à 10 14 35](https://github.com/iciamyplant/IA_vocales/assets/57531966/6cb3f575-8bd1-40e1-adc3-cee903e50c0a)

Fonctionnement text to speech : 
En général pour synthétiser une voix avec des modèles de Deep Learning (sans forcément vouloir une voix spécifique) on fait correspondre un signal audio (transformé en spectrogramme) à un texte. Le but est globalement de faire correspondre le texte à des phonèmes, puis chaque phonème à une suite de bouts de spectrogrammes dans un modèle acoustique optimisé. 

- input text
- normalization
- text processing
- phonème
- acoustic model
- waveform blocks
- speech waveform blocks

![Capture d’écran 2023-11-14 à 10 58 27](https://github.com/iciamyplant/IA_vocales/assets/57531966/3c2ee694-53f0-4c71-b477-42f58d86caaf)

### II.4 Remarque

Quelque chose qui est assez important à savoir aussi, c’est que dans les faits on entraine un énorme modèle avec plein plein de données de plein de speakers différents pour bien pouvoir généraliser à plein de nouvelles personnes, et derrière soit on utilise directement cet encoder général soit on ”fine-tune” avec nos propres données (on ré-entraine avec nos données pour optimiser encore une fois). 

Globalement la synthèse vocale est quelque chose qui existe déjà depuis un moment, ce qui est ”nouveau” c’est le clonage. Du coup il existe plein de librairies pour faire du Text-to-Speech 



# III - RVC & Hugging Face

Retrieval-based Voice Conversion : fournit toute une pipeline qui te permet de fine tuner leur modèles sur n’importe quel dataset.
- discords AIHub & AI Hub France
- documentation AI Hub France : https://docs.aihubfrance.fr/

### III.1 Utiliser un modèle pré-entraîné avec Hugging Face

Hugging face = plateforme et communauté open-source tournée vers le ML et de la science des données. Sur Hugging Face, les utilisateurs peuvent créer, déployer et entraîner des modèles de ML. Hugging Face héberge des milliers de modèles de ML, datasets et démos. Donc on peut voir et utiliser le code derrière les modèles (contrairement à Bard ou Chatgpt). Hugging Face a également un classement public qui suit, classe et évalue les LLM et chatbots qui sont sur la plateforme, computer vision models, audio models, image models..

Spaces = moyen d'héberger des app de démonstration ML directement sur votre profil. Permet qu'à partir d'un code, créer une app autour de ce code et partager l'app sur son profil en quelques minutes. Pour build cette app, Hugging Face travaille avec deux librairies opensource : Gradio et Streamlit. On choisit une des deux librairies, on donne notre nom à notre space, et on peut déposer notre code, et créer une app. [explication of spaces](https://huggingface.co/spaces/launch). Maintenant il suffit de donner le link de notre space pour partager notre modèle.

Gradio (SDK) = Create interactive ML demos with just a few lines of Python. Use your own models or existing HF models powered by the Inference API. Gradio est le moyen le plus rapide de démontrer votre modèle de ML avec une interface Web conviviale afin que tout le monde puisse l'utiliser, n'importe où



[Tutoriel créé par AI Hub France](https://docs.aihubfrance.fr/guides-clone-de-voix/hugging-face)

##### Préparation du Space

1. Pour commencer aller sur Hugging Face et connectez-vous à votre compte.
2. Ensuite, accédez au [Space RVC](https://huggingface.co/spaces/Clebersla/RVC_V2_Huggingface_Version) ==> lien fonctionne pas, je vais test avec ça : [lien](https://huggingface.co/spaces/TheStinger/Ilaria_RVC)
3. Cliquez sur les trois petits points en haut à droite
4. Puis sur "Duplicate this Space" (Dupliquer cet espace)
5. Ne changez rien et cliquez sur "Duplicate Space" (Dupliquer l'espace)
6. Ensuite, cliquez sur la croix pour fermer le terminal (cmd)
7. Enfin, attendez que Gradio s'affiche :

##### Ajout de modèle

[Tuto pour dwld modèle](https://docs.aihubfrance.fr/modeles/recherche-ton-modele)

Trouver son modèle sur [Weights](https://www.weights.gg/fr), site qui répertorie tous les modèles RVC du serveur Discord AI HUB et AI HUB FRANCE, disponibles gratuitement (modèles du discord synchro direct sur weights)




### III.2 Entraîner mon propre modèle avec RVC

[RVC Tutorial](https://www.youtube.com/watch?v=9TroP5mR3CM)
[another RVC tutorial](https://www.youtube.com/watch?v=9TroP5mR3CM)





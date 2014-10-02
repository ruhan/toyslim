"""
SLIM basic implementation. To understand deeply how it works we encourage you to
read "SLIM: Sparse LInear Methods for Top-N Recommender Systems".
"""
from cdecimal import Decimal
import numpy as np
from util import tsv_to_matrix
from metrics import compute_precision
from scipy.sparse import lil_matrix
from slim import slim_train
from scipy.sparse import csr_matrix
import subprocess, shlex
import os
import json

class LDA:
    """
    Generate a LDA model that integrates cidades x users x atractions in the
    same latent factor space.
    """
    def __init__(self, B, topics=20):
        self.model = {}
        self.generate_lda_model(B, topics)

    def generate_lda_model(self, B, topics, beta=0.01):
        """
        Executes LDA generating LDA Model
        """
        # Suggestions from plda:
        # https://code.google.com/p/plda/wiki/PLDAQuickStart
        alpha = 50/float(topics)

        # Se o modelo LDA ainda nao existe para o fold, inferi-lo
        path_base = './exp/'
        path_base_lda = '%slda/' % path_base
        path_test_data = '%stest_data.txt' % (path_base_lda)

        m, n = B.shape
        try:
            os.makedirs(path_base_lda)
        except OSError:
            pass

        results = []
        for user in range(1, m):
            attractions = list(B[user].nonzero()[0])
            attractions.append("") # para colocar o ultimo 1
            attractions = [ str(a) for a in attractions ]
            results.append(" 1 ".join(attractions))

        a = open(path_test_data, 'w')
        a.write("\n".join(results))
        a.close()

        comando_modelo = (
            '../plda/lda --num_topics %(topics)s --alpha %(alpha)s --beta %(beta)s '
            '--training_data_file %(path)stest_data.txt '
            '--model_file %(path)slda_model.txt --burn_in_iterations 250 '
            '--total_iterations 300'
        ) % {'path': path_base_lda, 'beta': beta, 'alpha': alpha, 'topics': topics}

        print comando_modelo
        output = subprocess.check_output(shlex.split(comando_modelo),
                                        stderr=subprocess.STDOUT)
        print output

        comando_inferencia = (
            '../plda/infer --alpha %(alpha)s --beta %(beta)s '
            '--inference_data_file %(path)stest_data.txt '
            '--inference_result_file %(path)sinference_result.txt '
            '--model_file %(path)slda_model.txt --total_iterations 300 '
            '--burn_in_iterations 250'
        ) % {'path': path_base_lda, 'beta': beta, 'alpha': alpha}

        print comando_inferencia
        output = subprocess.check_output(shlex.split(comando_inferencia),
                                        stderr=subprocess.STDOUT)
        print output

        # Handling LDA for attractions
        lda_attractions = {}
        for l in open('%slda_model.txt' % path_base_lda):
            attraction, data = l.split('\t')
            data = data.split(' ')
            data = [ Decimal(d) for d in data ]
            s = sum(data)
            data = [ d/s for d in data ]
            lda_attractions[attraction] = data

        # Handling LDA for each user
        lda_users = {}
        user = 0
        for l in open('%sinference_result.txt' % path_base_lda):
            user += 1
            data = l.split(' ')
            data = [ Decimal(d) for d in data ]
            s = sum(data)
            data = [ d/s for d in data ]
            lda_users[user] = data

        self.model = {
            'users': lda_users,
            'attractions': lda_attractions,
        }

class LDAHierarquical(object):
    """
    Infering the third latent features (cities) from its hierarquical structure
    of attractions.
    """
    def __init__(self, B, hierarchy, topics=20):
        self.lda = LDA(B, topics=topics)
        self.hierarchy = hierarchy
        self.model = self.fill_hierarchical_model(self.lda.model, hierarchy, topics)

    def fill_hierarchical_model(self, model, hierarchy, topics):
        """
        Initially we fill it using mean, in the future we will improve it using
        some statistical theory.
        """
        # TODO: in the future, it should be generated automatically by slim, as
        # the first step of all algorithms
        normalized = json.loads(open('data/normalized.json').read())
        normalized_side_information = normalized['atracoes']
        inverse_normalized_side = { v:k for k, v in normalized_side_information.iteritems() }

        # Inversing hierarchy to simplify our work
        inverse_hierarchy = {}

        for city, attractions in hierarchy.iteritems():
            for attraction in attractions:
                if not attraction in inverse_hierarchy:
                    inverse_hierarchy[attraction] = city

        # Organizing cities
        cities = {}
        attractions = set(model['attractions'].keys())
        attractions_denormalized = [ inverse_normalized_side[int(at)] for at in attractions ]

        for attraction in attractions_denormalized:
            city = inverse_hierarchy[int(attraction)]
            # There are some attractions that are not related to the cities
            # that we have seen in SLIM data (users x cities). These cities are
            # not useful to us, so we will ignore them
            try:
                city_normalized = normalized['cidades'][city]
            except KeyError:
                pass

            if city_normalized in cities:
                cities[city_normalized].append(normalized_side_information[attraction])
            else:
                cities[city_normalized] = [normalized_side_information[attraction]]

        lda_cities = {}
        # Creating the model for the cities that we will really use in the
        # algorithm. Note that hierarchy can contain all cities or only which
        # we will use
        for city, attractions in cities.iteritems():
            lda_city = [0] * topics

            for attraction in attractions:
                for i in range(topics):
                    lda_city[i] += model['attractions'][str(attraction)][i]

            for i in range(topics):
                lda_city[i] /= len(attractions)

            lda_cities[city] = lda_city

        model['cities'] = lda_cities
        return model

def slim_lda_recommender(A, W, lda):
    """
    Generate the A_hat recommendations matrix
    """
    # XXX: I don't know why, but we need to use csr conversion here. Without it
    # we are receiving a ValueError
    A_hat = csr_matrix(A) * W

    recommendations = {}
    m, n = A.shape

    LDA_matrix = lil_matrix((m, n))

    for u in range(1, m):
        for i in range(1, n):
            lda_u = lda.model['users'][u]
            lda_i = lda.model['cities'].get(i, None)

            # As attractions data is sparse, it is possible that our inferred
            # lda to cities hasn't all cities
            if lda_u and lda_i:
                lda_u = np.array(lda_u)
                lda_i = np.array(lda_i)
                dot_v = np.dot(lda_u, lda_i)
                LDA_matrix[u, i] = dot_v
            else:
                LDA_matrix[u, i] = 0

    LDA_matrix = LDA_matrix.toarray()
    LDA_matrix = LDA_matrix/(np.max(LDA_matrix)-np.min(LDA_matrix[LDA_matrix>0]))
    #from util import show_matplot_fig
    #import pdb;pdb.set_trace()
    #sorted(list(LDA_matrix[1][LDA_matrix[1]>0]/med))

    # Organizing A_hat matrix to simplify Top-N recommendation
    for u in range(1, m):
        for i in range(1, n):
            v = A_hat[u][i] * LDA_matrix[u, i]

            if v > 0:
                # NOTE: it only recommends items that the user haven't rated yet
                # Because that, we ignore already rated items
                if A[u][i] == 0:
                    if u not in recommendations:
                        recommendations[u] = [(i, v)]
                    else:
                        recommendations[u].append((i, v))

    # Ordering the most probable recommendations by A_hat
    for u in recommendations.iterkeys():
        recommendations[u].sort(reverse=True, cmp=lambda x, y: cmp(x[1], y[1]))

    # Removing W training weights of our recommendations
    for u in recommendations:
        for i, t in enumerate(recommendations[u]):
            recommendations[u][i] = t[0]

    return recommendations


def main(train_file, user_item_side_information_file, hierarchy_file, test_file):
    A = tsv_to_matrix(train_file)
    B = tsv_to_matrix(user_item_side_information_file)
    hierarchy = json.loads(open(hierarchy_file).read())

    lda = LDAHierarquical(B, hierarchy, topics=15)

    #####REMOVE_IT
    def important_topics(x, topics):
        if not x:
            return x
        transf = [ (i, j) for i, j in enumerate(x) ]
        transf = sorted(transf, cmp=lambda x, y: cmp(x[1], y[1]))
        return [ i[0] for i in transf[:topics] ]

    topics = 3

    coincidencias = []
    for user in range(1, 101):
        # Topicos do usuario 10
        user_topics = important_topics(lda.model['users'][user], topics)

        # Topicos das cidades de teste do usuario 10
        T = tsv_to_matrix(test_file)
        cities = T[user].nonzero()[0]

        cities_topics = [ important_topics(lda.model['cities'].get(city, []), topics) for city in cities ]


        total = 0
        topics_compared = 0
        coinc = 0
        for city_topic in cities_topics:
            if city_topic:
                coinc += len(set(user_topics) & set(city_topic))
                topics_compared += len(user_topics)
                total += 1
            else:
                pass

        if total:
            perc = (coinc/float(topics_compared))
        else:
            perc = -1

        coincidencias.append([coinc, topics_compared, perc])

    aa = open('/tmp/coincidencias.json', 'w')
    aa.write(json.dumps(coincidencias))
    aa.close()
    #####

    W = slim_train(A)

    recommendations = slim_lda_recommender(A, W, lda)

    compute_precision(recommendations, test_file)

if __name__ == '__main__':
    main('data/train_100.tsv', 'data/user_item_side_information_100.tsv',
         'data/hierarchy.json', 'data/test_100.tsv', )

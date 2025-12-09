## STENCIL

## Introduction

Lors de cette partie du projet, différentes versions ont été implémentées, chaque type d'implémentation est associé à son propre fichier.
Pour avoir l'exécutable de chacun d'eux, il suffit de faire make nom_du_fichier (ex : make stencil_vec). Ensuite, au sein de ces fichiers, différentes implémentations sont proposées et pour pouvoir tester chacune d'elles, il suffit de changer la fonction appelée dans la fonction run. De plus, si vous compilez la partie stencil du projet, un fichier load-modules est à votre disposition en faisant source load-modules.


## Stencil séquentiel

Pour la partie séquentielle, seule une fonction a été faite, la fonction naive_stencil_func_good_order() qui reprend la fonction naive et remet les boucles dans l'ordre pour avoir des accès contigus en mémoire. Dans l'idée, nous voulions aussi faire en sorte d'éviter la copie et plutôt échanger les pointeurs du tableau temporaire avec Mesh, mais cela ne fonctionne pas, les tests demandent bien une écriture sur le tableau Mesh. Donc cette implémentation avec le test basique qui lance 10 runs avec chacune d'elles 100 répétitions et des configurations différentes, on a des temps entre 0,63 s et 0,65 s.

## Stencil vectorielle

Sur la partie vectorielle, il y a deux fonctions : la première qui vectorise la partie calcul et la partie copie et la deuxième fonction fait la même chose à une exception, on initialise un tableau de valeurs vectorisées pour les coefs avant les boucles, ce qui permet d'éviter d'aller récupérer les valeurs de chaque coef à chaque fois. Cela permet de gagner un peu de temps. Ici nous serons à peine plus rapides que la version séquentielle, nous serons entre 0,60 s et 0,62 s sur le même test. Cela peut s'expliquer par le fait qu'on compile avec O3, donc, là, lorsqu'on met les boucles dans le bon ordre, le compilateur avait déjà vectorisé le code.

## Stencil starpu

Dans cette partie, nous avons implémenté quatre fonctions différentes. la première starpu_stencil_func() qui construit une tâche par indice du tableau, ce qui n'est pas performant du tout, les tests étaient très longs étant donné que pour chaque tâche on faisait un data_register de tout le tableau. C'est pourquoi nous sommes passés à la deuxième version starpu_stencil_func_v2() qui, elle, fait une tâche par ligne, ce qui est déjà plus rapide. La troisième version, elle, utilise le partitionnement starpu_stencil_func_v2_partitioned(). Pour cela on a gardé l'implémentation de la version 2 et on a un bon gain de temps, mais on a un temps entre 12 s et 13 s, ce qui est moins bien que le séquentiel. Nous avons donc essayé de faire une quatrième version avec des plus gros blocs pour réduire le temps passé à faire des data_register, mais nous n'avons pas réussi à garder le partitionnement. Sur cette quatrième version starpu_stencil_func_v2_big_tasks(), on est entre 11,9 s et 12,2 s, ce qui est un peu mieux mais pas de beaucoup. Néanmoins on voit que cette version dépasse un peu la version 3 alors qu'on utilise pas le partitionnement, donc on peut supposer qu'avec le partitionnement sur cette fonction on pourrait avoir une fonction plus rapide que le séquentiel. Lors de nos différents tests, nous avons pu voir que les CPU n'étaient pas du tout utilisés à leur plein potentiel, ce qui est dû à un temps de calcul trop faible par rapport au reste.


## Stencil starpu et vec

Ici nous aurons fait deux versions, la première starpu_vec_stencil_func() qui rajoute juste les calculs vectoriels à la version big_task de la partie précédente. Sur cette version, on a presque un x 2 sur le temps, on est entre 7,4 s et 7,6 s. Nous avons donc voulu tester d'implémenter une deuxième version starpu_vec_stencil_func_v2() qui utilise le partitionnement, mais nous n'avons pas pu finir cette implémentation, elle est donc non fonctionnelle.
# 3_Matrix_Multiplication

Подробнее про метод можно прочитать [здесь](https://ru.wikipedia.org/wiki/Алгоритм_умножения_матриц#Алгоритм_Разделяй-и-властвуй).

В репозитории две программы, решающие эту задачу: 
+ ul в main.c вы можете найти непараллельное решение;
+ ul в 3mm_block.c вы можете найти параллельное решение, используя библиотеку MPI. 

Программа тестировалась на суперкомпьютере МГУ IBM Polus. Результаты описаны в файле results.pdf.

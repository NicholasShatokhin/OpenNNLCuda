#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <ctime>

inline void initialize_random_generator()
{
    srand(time(NULL));
}

/*float closed_interval_rand(float x0, float x1)
{
    return x0 + (x1 - x0) * rand() / ((float) RAND_MAX);
}*/

/*float unified_random()
{
    return closed_interval_rand(0, 2) - 1;
}*/

#endif // UTILS_H

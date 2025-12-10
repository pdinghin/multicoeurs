#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <immintrin.h>
#include "starpu.h"

#define ELEMENT_TYPE float
#define NB_ELEMENT_VECT2 32/sizeof(ELEMENT_TYPE)

#define DEFAULT_MESH_WIDTH 2000
#define DEFAULT_MESH_HEIGHT 1000
#define DEFAULT_NB_ITERATIONS 100
#define DEFAULT_NB_REPEAT 10

#define STENCIL_WIDTH 3
#define STENCIL_HEIGHT 3

#define TOP_BOUNDARY_VALUE 10
#define BOTTOM_BOUNDARY_VALUE 5
#define LEFT_BOUNDARY_VALUE -10
#define RIGHT_BOUNDARY_VALUE -5

#define MAX_DISPLAY_COLUMNS 20
#define MAX_DISPLAY_LINES 100

#define EPSILON 1e-3

static const ELEMENT_TYPE stencil_coefs[STENCIL_HEIGHT * STENCIL_WIDTH] =
    {
        0.25 / 3,  0.50 / 3, 0.25 / 3,
        0.50 / 3, -1.00,     0.50 / 3,
        0.25 / 3,  0.50 / 3, 0.25 / 3};

enum e_initial_mesh_type
{
        initial_mesh_zero = 1,
        initial_mesh_random = 2
};

struct s_settings
{
        int mesh_width;
        int mesh_height;
        enum e_initial_mesh_type initial_mesh_type;
        int nb_iterations;
        int nb_repeat;
        int enable_output;
        int enable_verbose;
};

struct starpu_parameters
{
        int stencil_widht;
        int stencil_height;
        int mesh_width;
        int mesh_height;
        int actual_x;
        int actual_y;
        int block_start;
        int block_end;
};

#define PRINT_ERROR(MSG)                                                    \
        do                                                                  \
        {                                                                   \
                fprintf(stderr, "%s:%d - %s\n", __FILE__, __LINE__, (MSG)); \
                exit(EXIT_FAILURE);                                         \
        } while (0)

#define IO_CHECK(OP, RET)                   \
        do                                  \
        {                                   \
                if ((RET) < 0)              \
                {                           \
                        perror((OP));       \
                        exit(EXIT_FAILURE); \
                }                           \
        } while (0)

static void usage(void)
{
        fprintf(stderr, "usage: stencil [OPTIONS...]\n");
        fprintf(stderr, "    --mesh-width  MESH_WIDTH\n");
        fprintf(stderr, "    --mesh-height MESH_HEIGHT\n");
        fprintf(stderr, "    --initial-mesh <zero|random>\n");
        fprintf(stderr, "    --nb-iterations NB_ITERATIONS\n");
        fprintf(stderr, "    --nb-repeat NB_REPEAT\n");
        fprintf(stderr, "    --output\n");
        fprintf(stderr, "    --verbose\n");
        fprintf(stderr, "\n");
        exit(EXIT_FAILURE);
}

static void init_settings(struct s_settings **pp_settings)
{
        assert(*pp_settings == NULL);
        struct s_settings *p_settings = calloc(1, sizeof(*p_settings));
        if (p_settings == NULL)
        {
                PRINT_ERROR("memory allocation failed");
        }
        p_settings->mesh_width = DEFAULT_MESH_WIDTH;
        p_settings->mesh_height = DEFAULT_MESH_HEIGHT;
        p_settings->initial_mesh_type = initial_mesh_zero;
        p_settings->nb_iterations = DEFAULT_NB_ITERATIONS;
        p_settings->nb_repeat = DEFAULT_NB_REPEAT;
        p_settings->enable_verbose = 0;
        p_settings->enable_output = 0;
        *pp_settings = p_settings;
}

static void parse_cmd_line(int argc, char *argv[], struct s_settings *p_settings)
{
        int i = 1;
        while (i < argc)
        {
                if (strcmp(argv[i], "--mesh-width") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < STENCIL_WIDTH)
                        {
                                fprintf(stderr, "invalid MESH_WIDTH argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->mesh_width = value;
                }
                else if (strcmp(argv[i], "--mesh-height") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < STENCIL_HEIGHT)
                        {
                                fprintf(stderr, "invalid MESH_HEIGHT argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->mesh_height = value;
                }
                else if (strcmp(argv[i], "--initial-mesh") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        if (strcmp(argv[i], "zero") == 0)
                        {
                                p_settings->initial_mesh_type = initial_mesh_zero;
                        }
                        else if (strcmp(argv[i], "random") == 0)
                        {
                                p_settings->initial_mesh_type = initial_mesh_random;
                        }
                        else
                        {
                                fprintf(stderr, "invalid initial mesh type\n");
                                exit(EXIT_FAILURE);
                        }
                }
                else if (strcmp(argv[i], "--nb-iterations") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < 1)
                        {
                                fprintf(stderr, "invalid NB_ITERATIONS argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->nb_iterations = value;
                }
                else if (strcmp(argv[i], "--nb-repeat") == 0)
                {
                        i++;
                        if (i >= argc)
                        {
                                usage();
                        }
                        int value = atoi(argv[i]);
                        if (value < 1)
                        {
                                fprintf(stderr, "invalid NB_REPEAT argument\n");
                                exit(EXIT_FAILURE);
                        }
                        p_settings->nb_repeat = value;
                }
                else if (strcmp(argv[i], "--output") == 0)
                {
                        p_settings->enable_output = 1;
                }
                else if (strcmp(argv[i], "--verbose") == 0)
                {
                        p_settings->enable_verbose = 1;
                }
                else
                {
                        usage();
                }

                i++;
        }

        if (p_settings->enable_output)
        {
                p_settings->nb_repeat = 1;
                if (p_settings->nb_iterations > 100)
                {
                        p_settings->nb_iterations = 100;
                }
        }
}

static void delete_settings(struct s_settings **pp_settings)
{
        assert(*pp_settings != NULL);
        free(*pp_settings);
        pp_settings = NULL;
}

static void allocate_mesh(ELEMENT_TYPE **pp_mesh, struct s_settings *p_settings)
{
        assert(*pp_mesh == NULL);
        ELEMENT_TYPE *p_mesh = calloc(p_settings->mesh_width * p_settings->mesh_height, sizeof(*p_mesh));
        if (p_mesh == NULL)
        {
                PRINT_ERROR("memory allocation failed");
        }
        *pp_mesh = p_mesh;
}

static void delete_mesh(ELEMENT_TYPE **pp_mesh)
{
        assert(*pp_mesh != NULL);
        free(*pp_mesh);
        pp_mesh = NULL;
}

static void init_mesh_zero(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        const int margin_x = (STENCIL_WIDTH - 1) / 2;
        const int margin_y = (STENCIL_HEIGHT - 1) / 2;
        int x;
        int y;
        for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
        {
                for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
                {
                        p_mesh[y * p_settings->mesh_width + x] = 0;
                }
        }
}

static void init_mesh_random(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        const int margin_x = (STENCIL_WIDTH - 1) / 2;
        const int margin_y = (STENCIL_HEIGHT - 1) / 2;
        int x;
        int y;
        for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
        {
                for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
                {
                        ELEMENT_TYPE value = rand() / (ELEMENT_TYPE)RAND_MAX * 20 - 10;
                        p_mesh[y * p_settings->mesh_width + x] = value;
                }
        }
}
static void init_mesh_values(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        switch (p_settings->initial_mesh_type)
        {
        case initial_mesh_zero:
                init_mesh_zero(p_mesh, p_settings);
                break;

        case initial_mesh_random:
                init_mesh_random(p_mesh, p_settings);
                break;

        default:
                PRINT_ERROR("invalid initial mesh type");
        }
}

static void copy_mesh(ELEMENT_TYPE *p_dst_mesh, const ELEMENT_TYPE *p_src_mesh, struct s_settings *p_settings)
{
        memcpy(p_dst_mesh, p_src_mesh, p_settings->mesh_width * p_settings->mesh_height * sizeof(*p_dst_mesh));
}

static void apply_boundary_conditions(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        const int margin_x = (STENCIL_WIDTH - 1) / 2;
        const int margin_y = (STENCIL_HEIGHT - 1) / 2;
        int x;
        int y;

        for (x = 0; x < p_settings->mesh_width; x++)
        {
                for (y = 0; y < margin_y; y++)
                {
                        p_mesh[y * p_settings->mesh_width + x] = TOP_BOUNDARY_VALUE;
                        p_mesh[(p_settings->mesh_height - 1 - y) * p_settings->mesh_width + x] = BOTTOM_BOUNDARY_VALUE;
                }
        }

        for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
        {
                for (x = 0; x < margin_x; x++)
                {
                        p_mesh[y * p_settings->mesh_width + x] = LEFT_BOUNDARY_VALUE;
                        p_mesh[y * p_settings->mesh_width + (p_settings->mesh_width - 1 - x)] = RIGHT_BOUNDARY_VALUE;
                }
        }
}

static void print_settings_csv_header(void)
{
        printf("mesh_width,mesh_height,nb_iterations,nb_repeat");
}

static void print_settings_csv(struct s_settings *p_settings)
{
        FILE *fptr;
        fptr = fopen("starpu_vec.csv", "a+");
        fprintf(fptr, "%d,%d,%d,%d,starpu_vec\n", p_settings->mesh_width, p_settings->mesh_height, p_settings->nb_iterations, p_settings->nb_repeat);
        fclose(fptr);
        //printf("%d,%d,%d,%d", p_settings->mesh_width, p_settings->mesh_height, p_settings->nb_iterations, p_settings->nb_repeat);
}

static void print_results_csv_header(void)
{
        printf("rep,timing,check_status");
}

static void print_results_csv(int rep, double timing_in_seconds, int check_status)
{
        printf("%d,%le,%d", rep, timing_in_seconds, check_status);
}

static void print_csv_header(void)
{
        print_settings_csv_header();
        printf(",");
        print_results_csv_header();
        printf("\n");
}

static void print_mesh(const ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        int x;
        int y;

        printf("[\n");
        for (y = 0; y < p_settings->mesh_height; y++)
        {
                if (y >= MAX_DISPLAY_LINES)
                {
                        printf("...\n");
                        break;
                }
                printf("[%03d: ", y);
                for (x = 0; x < p_settings->mesh_width; x++)
                {
                        if (x >= MAX_DISPLAY_COLUMNS)
                        {
                                printf("...");
                                break;
                        }
                        printf(" %+8.2lf", p_mesh[y * p_settings->mesh_width + x]);
                }
                printf("]\n");
        }
        printf("]");
}

static void write_mesh_to_file(FILE *file, const ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        int x;
        int y;
        int ret;

        for (y = 0; y < p_settings->mesh_height; y++)
        {
                for (x = 0; x < p_settings->mesh_width; x++)
                {
                        if (x > 0)
                        {
                                ret = fprintf(file, ",");
                                IO_CHECK("fprintf", ret);
                        }

                        ret = fprintf(file, "%lf", p_mesh[y * p_settings->mesh_width + x]);
                        IO_CHECK("fprintf", ret);
                }

                ret = fprintf(file, "\n");
                IO_CHECK("fprintf", ret);
        }
}


static void naive_stencil_func(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        const int margin_x = (STENCIL_WIDTH - 1) / 2;
        const int margin_y = (STENCIL_HEIGHT - 1) / 2;
        int x;
        int y;

        ELEMENT_TYPE *p_temporary_mesh = malloc(p_settings->mesh_width * p_settings->mesh_height * sizeof(*p_mesh));
        for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
        {
                for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
                {
                        ELEMENT_TYPE value = p_mesh[y * p_settings->mesh_width + x];
                        int stencil_x, stencil_y;
                        for (stencil_x = 0; stencil_x < STENCIL_WIDTH; stencil_x++)
                        {
                                for (stencil_y = 0; stencil_y < STENCIL_HEIGHT; stencil_y++)
                                {
                                        value +=
                                            p_mesh[(y + stencil_y - margin_y) * p_settings->mesh_width + (x + stencil_x - margin_x)] * stencil_coefs[stencil_y * STENCIL_WIDTH + stencil_x];
                                }
                        }
                        p_temporary_mesh[y * p_settings->mesh_width + x] = value;
                }
        }

        for (x = margin_x; x < p_settings->mesh_width - margin_x; x++)
        {
                for (y = margin_y; y < p_settings->mesh_height - margin_y; y++)
                {
                        p_mesh[y * p_settings->mesh_width + x] = p_temporary_mesh[y * p_settings->mesh_width + x];
                }
        }
}




void stencil_cpu_func_block(void *buffers[], void *cl_arg) {
    struct starpu_vector_interface *mesh_vh = buffers[0];
    struct starpu_vector_interface *tmp_vh  = buffers[1];
    struct starpu_vector_interface *coef_vh = buffers[2];

    ELEMENT_TYPE *mesh = (ELEMENT_TYPE *)STARPU_VECTOR_GET_PTR(mesh_vh);
    ELEMENT_TYPE *tmp  = (ELEMENT_TYPE *)STARPU_VECTOR_GET_PTR(tmp_vh);
    float *coefs       = (float *)STARPU_VECTOR_GET_PTR(coef_vh);

    struct starpu_parameters params;
    starpu_codelet_unpack_args(cl_arg, &params);

    int margin_x = (params.stencil_widht - 1) / 2;
    int margin_y = (params.stencil_height - 1) / 2;
    int mesh_w   = params.mesh_width;
    int stencil_y,stencil_x;    
    int x ;
    int stencil_w = params.stencil_widht;
    int stencil_h = params.stencil_height;
    __m256 tab_stencil_coef[STENCIL_WIDTH * STENCIL_HEIGHT] ;
    for(int i = 0 ; i < STENCIL_WIDTH * STENCIL_HEIGHT ; i++){
                tab_stencil_coef[i] = _mm256_set1_ps(stencil_coefs[i]);
    }
    for (int y = params.block_start; y < params.block_end; y++) {
        for (x = margin_x; x < mesh_w - margin_x; x+=8) {
           __m256 value = _mm256_loadu_ps((mesh + y * params.mesh_width + x));
                        
                        for(stencil_y = 0 ; stencil_y < stencil_h ; stencil_y++)
                        {
                                for(stencil_x = 0 ; stencil_x < stencil_w ; stencil_x++)
                                {       
                                        __m256 a = _mm256_loadu_ps((mesh + (y + stencil_y - margin_y) * params.mesh_width + (x + stencil_x - margin_x)));
                                        value = _mm256_fmadd_ps(a,tab_stencil_coef[stencil_y * stencil_w + stencil_x],value);
                                }
                        }
                        _mm256_storeu_ps(tmp + y * params.mesh_width + x ,value);
        }

        for( ;  x < params.mesh_width - margin_x ; x++)
                {
                     ELEMENT_TYPE value = mesh[y * params.mesh_width + x];
                        for (stencil_y = 0; stencil_y < stencil_h; stencil_y++)
                        {
                                for (stencil_x = 0; stencil_x < stencil_w; stencil_x++)
                                {
                                        value +=
                                            mesh[(y + stencil_y - margin_y) * params.mesh_width + (x + stencil_x - margin_x)] * coefs[stencil_y * stencil_w + stencil_x];
                                }
                        }
                        tmp[y * params.mesh_width + x] = value;   
                }
    }
}

void copy_block_cpu_func(void *buffers[], void *cl_arg) {
    struct starpu_vector_interface *mesh_vh = buffers[0]; /* write */
    ELEMENT_TYPE *mesh = (ELEMENT_TYPE *)STARPU_VECTOR_GET_PTR(mesh_vh);

    struct starpu_vector_interface *tmp_vh = buffers[1];  /* read */
    ELEMENT_TYPE *tmp  = (ELEMENT_TYPE *)STARPU_VECTOR_GET_PTR(tmp_vh);

    struct starpu_parameters params;
    starpu_codelet_unpack_args(cl_arg, &params);

    int margin_x = (params.stencil_widht - 1) / 2;
    int mesh_w   = params.mesh_width;
    int x;
    for (int y = params.block_start; y < params.block_end; y++) {
        for(x = margin_x; x <= params.mesh_width - margin_x - 8 ; x+=8)
        {
                __m256 value = _mm256_loadu_ps(tmp + y * params.mesh_width + x);
                _mm256_storeu_ps(mesh + y * params.mesh_width + x,value);
        }

        for( ;  x < params.mesh_width - margin_x ; x++)
        {
                mesh[y * params.mesh_width + x] = tmp[y * params.mesh_width + x];
        }
    }
}



/**
 * Fonction non fonctionnelle
 * L'objectif était de faire du partitionnement mais la fonction n'est pas finis d'implémenter.
 */
static void starpu_vec_stencil_func_v2(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
    const int mesh_w = p_settings->mesh_width;
    const int mesh_h = p_settings->mesh_height;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    int nb_threads = sysconf(_SC_NPROCESSORS_ONLN);

    ELEMENT_TYPE *p_temporary_mesh = malloc((size_t)mesh_w * mesh_h * sizeof(*p_temporary_mesh));
    if (!p_temporary_mesh) {
        fprintf(stderr, "malloc failed\n");
        return;
    }

    int ret = starpu_init(NULL);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");


    starpu_data_handle_t mesh_handle, temporary_handle, coef_handle;
    starpu_vector_data_register(&mesh_handle, STARPU_MAIN_RAM, (uintptr_t)p_mesh, mesh_w * mesh_h, sizeof(p_mesh[0]));
    starpu_vector_data_register(&temporary_handle, STARPU_MAIN_RAM, (uintptr_t)p_temporary_mesh, mesh_w * mesh_h, sizeof(p_temporary_mesh[0]));
    starpu_vector_data_register(&coef_handle, STARPU_MAIN_RAM, (uintptr_t)stencil_coefs, STENCIL_WIDTH * STENCIL_HEIGHT, sizeof(stencil_coefs[0]));


    struct starpu_data_filter filter = {
        .filter_func = starpu_vector_filter_block,
        .nchildren = nb_threads
    };
    starpu_data_partition(mesh_handle, &filter);
    starpu_data_partition(temporary_handle, &filter);


    struct starpu_codelet stencil_cl = {
        .cpu_funcs = { stencil_cpu_func_block },
        .nbuffers = 3,
        .modes = { STARPU_R, STARPU_W, STARPU_R }
    };

    struct starpu_codelet copy_cl = {
        .cpu_funcs = { copy_block_cpu_func },
        .nbuffers = 2,
        .modes = { STARPU_W, STARPU_R } 
    };

    int effective_height = mesh_h - 2 * margin_y;
    if (effective_height <= 0) effective_height = mesh_h; 
    int block_height = effective_height / nb_threads;
    if (block_height == 0) block_height = 1;


    for (int t = 0; t < nb_threads; ++t) {
        int block_start = margin_y + t * block_height;
        int block_end = block_start + block_height;
        if (t == nb_threads - 1) block_end = mesh_h - margin_y; 

      
        starpu_data_handle_t mesh_sub = starpu_data_get_sub_data(mesh_handle, 1, t);
        starpu_data_handle_t tmp_sub  = starpu_data_get_sub_data(temporary_handle, 1, t);


        struct starpu_parameters params;
        params.block_start = block_start;
        params.block_end   = block_end;
        params.mesh_width  = mesh_w;
        params.mesh_height = mesh_h;
        params.stencil_widht = STENCIL_WIDTH;
        params.stencil_height = STENCIL_HEIGHT;

        starpu_task_insert(&stencil_cl,
            STARPU_R, mesh_sub,   
            STARPU_W, tmp_sub,        
            STARPU_R, coef_handle,
            STARPU_VALUE, &params, sizeof(params),
            0);

        starpu_task_wait_for_all();
        starpu_task_insert(&copy_cl,
            STARPU_W, mesh_sub,
            STARPU_R, tmp_sub,
            STARPU_VALUE, &params, sizeof(params),
            0);

    }


    starpu_task_wait_for_all();


    starpu_data_unpartition(mesh_handle, 0);
    starpu_data_unpartition(temporary_handle, 0);

    starpu_data_unregister(mesh_handle);
    starpu_data_unregister(temporary_handle);
    starpu_data_unregister(coef_handle);

    starpu_shutdown();

    free(p_temporary_mesh);
}

/**Fonction qui utilise la version grosse tache de la partie starpu et ajoute les calculs vectoriels */
static void starpu_vec_stencil_func(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
    const int mesh_w = p_settings->mesh_width;
    const int mesh_h = p_settings->mesh_height;
    const int margin_y = (STENCIL_HEIGHT - 1) / 2;
    int nb_threads = sysconf(_SC_NPROCESSORS_ONLN);
    ELEMENT_TYPE *p_temporary_mesh = malloc(mesh_w * mesh_h * sizeof(*p_temporary_mesh));
    if (!p_temporary_mesh) {
        fprintf(stderr, "malloc failed\n");
        return;
    }

    int ret = starpu_init(NULL);
    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

    starpu_data_handle_t mesh_handle, temporary_handle, coef_handle;
    starpu_vector_data_register(&mesh_handle, STARPU_MAIN_RAM, (uintptr_t)p_mesh, mesh_w * mesh_h, sizeof(p_mesh[0]));
    starpu_vector_data_register(&temporary_handle, STARPU_MAIN_RAM, (uintptr_t)p_temporary_mesh, mesh_w * mesh_h, sizeof(p_temporary_mesh[0]));
    starpu_vector_data_register(&coef_handle, STARPU_MAIN_RAM, (uintptr_t)stencil_coefs, STENCIL_WIDTH * STENCIL_HEIGHT, sizeof(stencil_coefs[0]));

    int effective_height = mesh_h - 2 * margin_y;
    int block_height = effective_height / nb_threads;
    if (block_height == 0) block_height = 1; 

    struct starpu_codelet stencil_cl = {
        .cpu_funcs = { stencil_cpu_func_block },
        .nbuffers = 3,
        .modes = { STARPU_R, STARPU_W, STARPU_R } 
    };

    for (int t = 0; t < nb_threads; t++) {
        int y_start = margin_y + t * block_height;
        int y_end = y_start + block_height;
        if (t == nb_threads - 1) y_end = mesh_h - margin_y; 

        struct starpu_parameters params;
        params.actual_y = y_start;
        params.block_start = y_start;
        params.block_end = y_end;
        params.mesh_width = mesh_w;
        params.mesh_height = mesh_h;
        params.stencil_height = STENCIL_HEIGHT;
        params.stencil_widht = STENCIL_WIDTH;

        starpu_task_insert(&stencil_cl,
            STARPU_R, mesh_handle,
            STARPU_W, temporary_handle,
            STARPU_R, coef_handle,
            STARPU_VALUE, &params, sizeof(params),
            0);
    }

    starpu_task_wait_for_all();

    struct starpu_codelet copy_cl = {
    .cpu_funcs = { copy_block_cpu_func }, 
    .nbuffers = 2,
    .modes = { STARPU_W, STARPU_R }
    };


    for (int t = 0; t < nb_threads; t++) {
        int block_start = margin_y + t * block_height;
        int block_end   = block_start + block_height;
        if (t == nb_threads - 1) block_end = mesh_h - margin_y; 

        struct starpu_parameters params;
        params.block_start = block_start;
        params.block_end   = block_end;
        params.mesh_width  = mesh_w;
        params.stencil_widht = STENCIL_WIDTH; 

        starpu_task_insert(&copy_cl,
            STARPU_W, mesh_handle,
            STARPU_R, temporary_handle,
            STARPU_VALUE, &params, sizeof(params),
            0
        );
    }
    starpu_task_wait_for_all();

    starpu_data_unregister(mesh_handle);
    starpu_data_unregister(temporary_handle);
    starpu_data_unregister(coef_handle);

    starpu_shutdown();
    free(p_temporary_mesh);
}



static void run(ELEMENT_TYPE *p_mesh, struct s_settings *p_settings)
{
        int i;
        for (i = 0; i < p_settings->nb_iterations; i++)
        {
                starpu_vec_stencil_func(p_mesh, p_settings);

                if (p_settings->enable_output)
                {
                        char filename[32];
                        snprintf(filename, 32, "run_mesh_%03d.csv", i);
                        FILE *file = fopen(filename, "w");
                        if (file == NULL)
                        {
                                perror("fopen");
                                exit(EXIT_FAILURE);
                        }
                        write_mesh_to_file(file, p_mesh, p_settings);
                        fclose(file);
                }

                if (p_settings->enable_verbose)
                {
                        printf("mesh after iteration %d\n", i);
                        print_mesh(p_mesh, p_settings);
                        printf("\n\n");
                }
        }
}

static int check(const ELEMENT_TYPE *p_mesh, ELEMENT_TYPE *p_mesh_copy, struct s_settings *p_settings)
{
        int i;
        for (i = 0; i < p_settings->nb_iterations; i++)
        {
                naive_stencil_func(p_mesh_copy, p_settings);

                if (p_settings->enable_output)
                {
                        char filename[32];
                        snprintf(filename, 32, "check_mesh_%03d.csv", i);
                        FILE *file = fopen(filename, "w");
                        if (file == NULL)
                        {
                                perror("fopen");
                                exit(EXIT_FAILURE);
                        }
                        write_mesh_to_file(file, p_mesh_copy, p_settings);
                        fclose(file);
                }

                if (p_settings->enable_verbose)
                {
                        printf("check mesh after iteration %d\n", i);
                        print_mesh(p_mesh_copy, p_settings);
                        printf("\n\n");
                }
        }

        int check = 0;
        int x;
        int y;
        for (y = 0; y < p_settings->mesh_height; y++)
        {
                for (x = 0; x < p_settings->mesh_width; x++)
                {
                        ELEMENT_TYPE diff = fabs(p_mesh[y * p_settings->mesh_width + x] - p_mesh_copy[y * p_settings->mesh_width + x]);
                        if (diff > EPSILON)
                        {
                                fprintf(stderr, "check failed [x: %d, y: %d]: run = %lf, check = %lf\n", x, y,
                                        p_mesh[y * p_settings->mesh_width + x],
                                        p_mesh_copy[y * p_settings->mesh_width + x]);
                                check = 1;
                        }
                }
        }

        return check;
}

int main(int argc, char *argv[])
{
        struct s_settings *p_settings = NULL;

        init_settings(&p_settings);
        parse_cmd_line(argc, argv, p_settings);

        ELEMENT_TYPE *p_mesh = NULL;
        allocate_mesh(&p_mesh, p_settings);

        ELEMENT_TYPE *p_mesh_copy = NULL;
        allocate_mesh(&p_mesh_copy, p_settings);

        {
                if (!p_settings->enable_verbose)
                {
                        print_csv_header();
                }

                int rep;
                for (rep = 0; rep < p_settings->nb_repeat; rep++)
                {
                        if (p_settings->enable_verbose)
                        {
                                printf("repeat %d\n", rep);
                        }

                        init_mesh_values(p_mesh, p_settings);
                        apply_boundary_conditions(p_mesh, p_settings);
                        copy_mesh(p_mesh_copy, p_mesh, p_settings);

                        if (p_settings->enable_verbose)
                        {
                                printf("initial mesh\n");
                                print_mesh(p_mesh, p_settings);
                                printf("\n\n");
                        }

                        struct timespec timing_start, timing_end;
                        clock_gettime(CLOCK_MONOTONIC, &timing_start);
                        run(p_mesh, p_settings);
                        clock_gettime(CLOCK_MONOTONIC, &timing_end);
                        double timing_in_seconds = (timing_end.tv_sec - timing_start.tv_sec) + 1.0e-9 * (timing_end.tv_nsec - timing_start.tv_nsec);

                        int check_status = check(p_mesh, p_mesh_copy, p_settings);

                        if (p_settings->enable_verbose)
                        {
                                print_csv_header();
                        }
                        print_settings_csv(p_settings);
                        printf(",");
                        print_results_csv(rep, timing_in_seconds, check_status);
                        printf("\n");
                }
        }

        delete_mesh(&p_mesh_copy);
        delete_mesh(&p_mesh);
        delete_settings(&p_settings);

        return 0;
}

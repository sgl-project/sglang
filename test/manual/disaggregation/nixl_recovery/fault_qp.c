/* Test-only RC queue-pair failure injection. Link with -ldl -lpthread, NOT
 * -libverbs: loading system verbs alongside a wheel's bundled verbs/libnl can
 * cause symbol collisions. Use only in an isolated fault-injection deployment.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <infiniband/verbs.h>
#include <link.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static void *verbs;
static int capture, count;
static struct ibv_qp *qps[4096];

static int find_verbs(struct dl_phdr_info *info, size_t size, void *data) {
    if (strstr(info->dlpi_name, "libibverbs-")) {
        verbs = dlopen(info->dlpi_name, RTLD_NOW | RTLD_NOLOAD);
        return verbs != NULL;
    }
    return 0;
}

static void init_verbs(void) {
    dl_iterate_phdr(find_verbs, NULL);
    if (!verbs) verbs = dlopen("libibverbs.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!verbs) { fprintf(stderr, "No loaded verbs library\n"); abort(); }
}

static void *symbol(const char *name) {
    pthread_once(&once, init_verbs);
    void *p = dlsym(verbs, name);
    if (!p) { fprintf(stderr, "Missing verbs symbol %s\n", name); abort(); }
    return p;
}

struct ibv_qp *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *attr) {
    struct ibv_qp *(*real)(struct ibv_pd *, struct ibv_qp_init_attr *) = symbol("ibv_create_qp");
    struct ibv_qp *qp = real(pd, attr);
    pthread_mutex_lock(&lock);
    if (qp && capture && qp->qp_type == IBV_QPT_RC && count < 4096) qps[count++] = qp;
    pthread_mutex_unlock(&lock);
    return qp;
}

int ibv_destroy_qp(struct ibv_qp *qp) {
    int (*real)(struct ibv_qp *) = symbol("ibv_destroy_qp");
    pthread_mutex_lock(&lock);
    for (int i = 0; i < count; i++) if (qps[i] == qp) qps[i] = NULL;
    pthread_mutex_unlock(&lock);
    return real(qp);
}

void fault_capture(int enabled) {
    pthread_mutex_lock(&lock);
    capture = enabled;
    if (enabled) count = 0;
    pthread_mutex_unlock(&lock);
}

int fault_count(void) { return count; }

/* Keep one captured QP intact to exercise a partial connection failure. */
int fault_disconnect_one(void) {
    int changed = 0;
    int (*modify)(struct ibv_qp *, struct ibv_qp_attr *, int) = symbol("ibv_modify_qp");
    pthread_mutex_lock(&lock);
    for (int i = 0; i < count - 1; i++) {
        if (!qps[i]) continue;
        struct ibv_qp_attr attr = {.qp_state = IBV_QPS_ERR};
        int rc = modify(qps[i], &attr, IBV_QP_STATE);
        fprintf(stderr, "FAULT_DISCONNECT qp=%u rc=%d\n", qps[i]->qp_num, rc);
        if (!rc) changed++;
    }
    pthread_mutex_unlock(&lock);
    return changed;
}

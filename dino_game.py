# python game with pygame : Jumping dino
# by. BlockDMask
import pygame
import sys
import pygame.font
# step1 : set screen, fps
# step2 : show dino, jump dino
# step3 : show tree, move tree

pygame.init()
pygame.display.set_caption('Jumping dino')
pygame.font.init()  # 폰트 초기화 (필요한 경우)
font = pygame.font.SysFont("Courier New", 20)
MAX_WIDTH = 800
MAX_HEIGHT = 400


def main():
    # set screen, fps
    screen = pygame.display.set_mode((MAX_WIDTH, MAX_HEIGHT))
    fps = pygame.time.Clock()

    # dino
    imgDino1 = pygame.image.load('images/dino1.png')
    imgDino2 = pygame.image.load('images/dino2.png')
    imgDino3 = pygame.image.load('images/dinodie.png')
    imgDino1= pygame.transform.scale(imgDino1, (imgDino1.get_width()*3, imgDino1.get_height()*3))
    imgDino2= pygame.transform.scale(imgDino2, (imgDino2.get_width()*3, imgDino2.get_height()*3))
    imgDino3= pygame.transform.scale(imgDino3, (imgDino3.get_width()*3, imgDino3.get_height()*3))
    dino_height = imgDino1.get_size()[1]
    dino_bottom = MAX_HEIGHT - dino_height
    dino_x = 50
    dino_y = dino_bottom
    jump_top = 200
    leg_swap = True
    is_bottom = True
    is_go_up = False

    # tree
    imgTree = pygame.image.load('images/tree.png')
    tree_height = imgTree.get_size()[1]
    tree_x = MAX_WIDTH
    tree_y = MAX_HEIGHT - tree_height


    #score
    score = 0
    dino_rect = imgDino1.get_rect()  # 공룡의 충돌 영역
    tree_rect = imgTree.get_rect()  # 나무의 충돌 영역
    tree_rect.width -= 20 
    tree_rect.height -= 10  

    # 게임 상태 변수
    game_active = True  # 게임이 활성 상태인지 표시

    while True:
        screen.fill((255, 255, 255))

        if game_active:
            # 점수 증가
            score += 1
            score_text = font.render(f"Score: {score}", True, (0, 0, 0))
            screen.blit(score_text, (MAX_WIDTH - 150, 20))

            # 충돌 영역 업데이트
            dino_rect.x = dino_x
            dino_rect.y = dino_y
            tree_rect.x = tree_x
            tree_rect.y = tree_y

            # 충돌 감지
            if dino_rect.colliderect(tree_rect):
                screen.blit(imgDino3, (dino_x, dino_y))  # 충돌 시 이미지 변경
                pygame.display.update()  # 변경된 이미지 업데이트
                pygame.time.delay(2000)
                game_active = False

            # 이벤트 체크 및 점프 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and is_bottom:
                        is_go_up = True
                        is_bottom = False

            # 공룡 움직임 처리
            if is_go_up:
                dino_y -= 10.0
            elif not is_go_up and not is_bottom:
                dino_y += 10.0

            if is_go_up and dino_y <= jump_top:
                is_go_up = False

            if not is_bottom and dino_y >= dino_bottom:
                is_bottom = True
                dino_y = dino_bottom

            # 나무 움직임 처리
            tree_x -= 12.0
            if tree_x <= 0:
                tree_x = MAX_WIDTH

            # 공룡 및 나무 그리기
            if leg_swap:
                screen.blit(imgDino1, (dino_x, dino_y))
                leg_swap = False
            else:
                screen.blit(imgDino2, (dino_x, dino_y))
                leg_swap = True
            screen.blit(imgTree, (tree_x, tree_y))

        else:
            # 게임 오버 화면 및 점수 표시
            game_over_text = font.render("Game Over", True, (0, 0, 0))
            screen.blit(game_over_text, (MAX_WIDTH // 2 - 50, MAX_HEIGHT // 2-40))
            final_score_text = font.render(f"Final Score: {score}", True, (0, 0, 0))
            screen.blit(final_score_text, (MAX_WIDTH // 2 - 60, MAX_HEIGHT // 2-10))
            restart_text = font.render("If you want RESTART, Press R!", True, (255,0,0))
            screen.blit(restart_text, (MAX_WIDTH // 2-170, MAX_HEIGHT // 2+20))
            
            # 게임 오버 상태에서의 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        # 게임 재시작
                        game_active = True
                        score = 0
                        dino_y = dino_bottom
                        tree_x = MAX_WIDTH
                        is_bottom = True
                        is_go_up = False

        # 화면 업데이트
        pygame.display.update()
        fps.tick(30)


if __name__ == '__main__':
    main()
